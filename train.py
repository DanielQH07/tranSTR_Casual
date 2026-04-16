# from torch.nn.modules.module import _IncompatibleKeys
import torch
import math
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
import os
from utils.logger import logger
import eval_mc
import time
import logging
import argparse
import os.path as osp
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from networks.model import VideoQAmodel
# from dataloader.dataset import VidQADataset 
from DataLoader import VideoQADataset

# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))
# torch.autograd.set_detect_anomaly(True)


def _masked_mean(tokens, mask):
    # tokens: [B, L, D], mask: [B, L] with True for valid tokens
    mask_f = mask.float().unsqueeze(-1)
    token_sum = (tokens * mask_f).sum(dim=1)
    token_count = mask_f.sum(dim=1).clamp_min(1.0)
    return token_sum / token_count


def _build_optimizer(model, args):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found for current stage settings.")

    non_text = [p for n, p in trainable if "text_encoder" not in n]
    text = [p for n, p in trainable if "text_encoder" in n]
    param_dicts = []
    if non_text:
        param_dicts.append({"params": non_text})
    if text:
        param_dicts.append({"params": text, "lr": args.text_encoder_lr})
    return torch.optim.AdamW(params=param_dicts, lr=args.lr, weight_decay=args.decay)


def _apply_stage_freeze(model, args, epoch):
    if args.stage_mode == "stage1_chain":
        for name, p in model.named_parameters():
            p.requires_grad = ("memory_mixer" in name)
        return

    if args.stage_mode == "stage2_qa" and args.stage2_freeze_backbone_epochs > 0:
        if epoch <= args.stage2_freeze_backbone_epochs:
            for name, p in model.named_parameters():
                train_head = (
                    name.startswith("ans_decoder")
                    or name.startswith("classifier")
                    or name.startswith("verifier")
                    or name.startswith("memory_mixer")
                )
                if "text_encoder" in name and not args.stage2_unfreeze_text_encoder:
                    p.requires_grad = False
                else:
                    p.requires_grad = train_head
            return

    # Default: train all params except optional text freeze flag.
    for name, p in model.named_parameters():
        if "text_encoder" in name and args.freeze_text_encoder and not args.stage2_unfreeze_text_encoder:
            p.requires_grad = False
        else:
            p.requires_grad = True


def train_stage1(model, optimizer, train_loader, device, args, use_amp=True, scaler=None):
    model.train()
    if not getattr(model, "enable_mixer", False):
        raise RuntimeError("Stage1 requires --enable_mixer.")

    total_step = max(1, len(train_loader))
    epoch_loss = 0.0
    valid_batches = 0

    for inputs in train_loader:
        if len(inputs) != 8:
            raise RuntimeError("stage1_chain currently expects raw-text batches (8 fields). Disable cached text features for stage1.")
        vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, _, chain_text, chain_valid = inputs
        vid_frame_feat = vid_frame_inputs.to(device)
        vid_obj_feat = vid_obj_inputs.to(device)
        ans_targets = ans_id.to(device)
        chain_valid = chain_valid.to(device).bool()

        with torch.amp.autocast('cuda', enabled=use_amp):
            aux = model(vid_frame_feat, vid_obj_feat, qns_w, ans_w, return_aux=True)
            mixed_mem = aux["mixed_mem"]  # [B, 5, L, D]
            mem_mask = aux["mem_mask"]    # [B, L]
            if mixed_mem is None:
                raise RuntimeError("Stage1 requires mixer-enabled mixed memory output.")

            # Pool each answer-conditioned memory branch.
            B, N, L, D = mixed_mem.shape
            mem_mask_exp = mem_mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, L)
            mem_pool = _masked_mean(mixed_mem.reshape(B * N, L, D), mem_mask_exp).reshape(B, N, D)

            # Chain text embedding is target; detach to keep Stage1 focused on mixer.
            with torch.no_grad():
                chain_embed = model.encode_text_global(list(chain_text), device)

            valid_idx = chain_valid.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                loss = aux["logits"].sum() * 0.0
            else:
                valid_mem_pool = mem_pool.index_select(0, valid_idx)
                valid_chain = chain_embed.index_select(0, valid_idx)
                valid_targets = ans_targets.index_select(0, valid_idx)

                pos_mem = valid_mem_pool[torch.arange(valid_mem_pool.size(0), device=device), valid_targets]
                s_all = F.cosine_similarity(valid_mem_pool, valid_chain.unsqueeze(1), dim=-1)  # [Bv, 5]
                s_pos = s_all.gather(1, valid_targets.unsqueeze(1)).squeeze(1)

                neg_mask = torch.ones_like(s_all, dtype=torch.bool)
                neg_mask.scatter_(1, valid_targets.unsqueeze(1), False)
                s_neg = s_all.masked_fill(~neg_mask, -1e9)
                max_neg = s_neg.max(dim=1).values

                align_loss = (1.0 - F.cosine_similarity(pos_mem, valid_chain, dim=-1)).mean()
                rank_loss = F.relu(args.stage1_rank_margin - s_pos + max_neg).mean()
                loss = args.stage1_align_weight * align_loss + args.stage1_rank_weight * rank_loss

                if args.stage1_use_qa_ce:
                    qa_loss = F.cross_entropy(aux["logits"].index_select(0, valid_idx), valid_targets)
                    loss = loss + args.stage1_qa_ce_weight * qa_loss

                valid_batches += 1

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    denom = valid_batches if valid_batches > 0 else total_step
    return epoch_loss / max(1, denom)


def train(model, optimizer, train_loader, xe, device, use_amp=True, scaler=None):
    model.train()
    total_step = len(train_loader)
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    for iter, inputs in enumerate(train_loader):
        # videos, qns_w, ans_w, ans_id, _ = inputs
        # video_inputs = videos.to(device)
        vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, _ = inputs
        vid_frame_feat = vid_frame_inputs.to(device)
        vid_obj_feat = vid_obj_inputs.to(device)
        ans_targets = ans_id.to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
             out_f = model( vid_frame_feat, vid_obj_feat, qns_w, ans_w)
             loss = xe(out_f, ans_targets)
        
        optimizer.zero_grad()
        if use_amp and scaler is not None:
             scaler.scale(loss).backward()
             scaler.step(optimizer)
             scaler.update()
        else:
             loss.backward()
             optimizer.step()
             
        epoch_loss += loss.item()
        prediction=out_f.max(-1)[1] # bs,
        prediction_list.append(prediction)
        answer_list.append(ans_id)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    
    return epoch_loss / total_step, acc_num*100.0 / len(ref_answers)
    

def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            # videos, qns_w, ans_w, ans_id, _ = inputs
            # video_inputs = videos.to(device)
            vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, _ = inputs
            vid_frame_feat = vid_frame_inputs.to(device)
            vid_obj_feat = vid_obj_inputs.to(device)
            out = model( vid_frame_feat, vid_obj_feat, qns_w, ans_w)
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(ans_id)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return acc_num*100.0 / len(ref_answers)


def predict(model,test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(test_loader):
            # videos, qns_w, ans_w, ans_id, qns_keys = inputs
            # video_inputs = videos.to(device)
            # qns_keys is passed in test_loader (see DataLoader logic in previous turns)
            # Need to ensure DataLoader returns 6 items for test split in the MAIN block too?
            # Yes, standard return is (frames, objs, q, a, id, keys)
            vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, qns_keys = inputs
            vid_frame_feat = vid_frame_inputs.to(device)
            vid_obj_feat = vid_obj_inputs.to(device)
            out = model(vid_frame_feat, vid_obj_feat, qns_w, ans_w)
            prediction=out.max(-1)[1] # bs,
            prediction_list.append(prediction)
            answer_list.append(ans_id)

            for qid, pred, ans in zip(qns_keys, prediction.data.cpu().numpy(), ans_id.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    
    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return results, acc_num*100.0 / len(ref_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train parameter")
    # general
    parser.add_argument("-v", type=str, required=True, help="version")
    parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=32)
    parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-5)
    parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=15)
    parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
    parser.add_argument("-es", action="store_true", help="early_stopping")
    parser.add_argument("-dropout", "-drop", type=float, help="dropout rate", default=0.1)
    parser.add_argument("-encoder_dropout", "-ep", type=float, help="dropout rate", default=0.1)   
    parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=1)
    # parser.add_argument("-mile_stone", "-mile", type=str, help="mile stone of MutiStepLr", default='7,10') 
    parser.add_argument("-gamma", "-ga", type=float, help="gamma of MultiStepLR", default=0.25)
    parser.add_argument("-decay", type=float, help="weight decay", default=0.001) 
    
    # dataset
    parser.add_argument('-dataset', default='causal-vid', choices=['msrvtt-qa', 'msvd-qa', 'next-qa', 'causal-vid'], type=str)
    parser.add_argument("-objs", default=20, type=int, help="sample of object feature")
    parser.add_argument("--sample_list_path", type=str, default=None,
                        help="Path to split pkl files (dataset-split-1)")
    parser.add_argument("--video_feature_path", type=str, default=None,
                        help="Path to visual features (appearance_feat.h5, motion_feat.h5)")
    parser.add_argument("--text_annotation_path", type=str, default=None,
                        help="Path to text annotations")
    parser.add_argument("--qtype", type=int, default=-1,
                        help="Question type for CausalVidQA: -1 for all, 0-5 for specific")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of videos for quick testing (e.g., 10, 50, 100)")
    parser.add_argument("--frame_feat_dim", type=int, default=4096,
                        help="Frame feature dimension (app+mot concatenated)")
    parser.add_argument("--obj_feat_dim", type=int, default=2053,
                        help="Object feature dimension (2048+5 bbox)")
    parser.add_argument("--grounding_dino_path", type=str, default=None,
                        help="Path to GroundingDINO feature files")
    
    # model
    parser.add_argument("-d_model", "-md",  type=int, help="hidden dim of vq encoder", default=768) 
    parser.add_argument("-word_dim", "-wd", type=int, help="word dim ", default=768)   
    parser.add_argument("-topK_frame", "-fk", type=int, help="word dim ", default=8)   
    parser.add_argument("-topK_obj", "-ok", type=int, help="word dim ", default=5)   
    parser.add_argument("-hard_eval", "-hd", action="store_true", help="hard selection during inference")
    
    # transformer
    parser.add_argument("-num_encoder_layers", "-el", type=int, help="number of encoder layers in transformer", default=1)
    parser.add_argument("-num_decoder_layers", "-dl", type=int, help="number of decoder layers in transformer", default=1)
    parser.add_argument("-n_query", type=int, help="num of query", default=5) 
    parser.add_argument("-nheads", type=int, help="num of attention head", default=8) 
    parser.add_argument("-normalize_before", action="store_true", help="pre or post normalize")
    parser.add_argument("-activation", default='relu', choices=['relu','gelu','glu'], type=str)
    
    
    # lan model
    parser.add_argument("-text_encoder_lr","-tlr", type=float, action="store", help="learning rate for lan model", default=5e-6)
    parser.add_argument("-freeze_text_encoder", action="store_true", help="freeze text encoder")
    parser.add_argument("-text_encoder_type", "-t", default="microsoft/deberta-base", choices=["roberta-base","distilroberta-base",\
                        "bert-base-uncased", "distilbert-base-uncased","microsoft/deberta-base",\
                            "microsoft/deberta-v3-base","microsoft/deberta-v3-small", "microsoft/deberta-v3-xsmall"], type=str)
    parser.add_argument('-text_pool_mode',"-pool", default=0, choices=[0,1,2],help="0last hidden, 1mean, 2max", type=int)
    
    # cl
    parser.add_argument("-pos_ratio", "-pr", type=float, help="postive ratio of fg token in trans decoder", default=0.7)   
    parser.add_argument("-neg_ratio", "-nr", type=float, help="negtive ratio of fg token in trans decoder", default=0.3) 
    parser.add_argument("-a", type=float, action="store", help="NCE loss multiplier", default=1) 
    
    # AMP setting (disable for DeBERTa FP16 issues)
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training (default: False)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")

    # Stage-wise training options
    parser.add_argument("--stage_mode", type=str, default="qa",
                        choices=["qa", "stage1_chain", "stage2_qa"],
                        help="qa: baseline CE, stage1_chain: train mixer with chain supervision, stage2_qa: QA finetune with stage1 weights")
    parser.add_argument("--enable_mixer", action="store_true",
                        help="Enable answer-conditioned CausalMemoryMixer")
    parser.add_argument("--mixer_hidden_dim", type=int, default=0,
                        help="Mixer MLP hidden dim (0 means 2*d_model)")
    parser.add_argument("--mixer_dropout", type=float, default=0.1,
                        help="Mixer dropout")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to stage1 checkpoint to load before stage2")
    parser.add_argument("--save_stage1_checkpoint", type=str, default=None,
                        help="Where to save stage1 checkpoint (default: ./models/stage1-<sign>.ckpt)")
    parser.add_argument("--qtype_subset", type=str, default="",
                        help="Comma-separated qtypes for subset training, e.g. explanatory,predictive,predictive_reason")
    parser.add_argument("--chain_data_root", type=str, default=None,
                        help="Root directory of chain supervision JSON files")
    parser.add_argument("--stage1_align_weight", type=float, default=1.0,
                        help="Weight for stage1 alignment loss")
    parser.add_argument("--stage1_rank_weight", type=float, default=0.2,
                        help="Weight for stage1 ranking loss")
    parser.add_argument("--stage1_rank_margin", type=float, default=0.2,
                        help="Margin for stage1 ranking loss")
    parser.add_argument("--stage1_use_qa_ce", action="store_true",
                        help="Optionally add QA CE term in stage1")
    parser.add_argument("--stage1_qa_ce_weight", type=float, default=0.1,
                        help="Weight for optional QA CE in stage1")
    parser.add_argument("--stage2_freeze_backbone_epochs", type=int, default=0,
                        help="Freeze backbone for first N epochs in stage2")
    parser.add_argument("--stage2_unfreeze_text_encoder", action="store_true",
                        help="Allow text encoder to unfreeze in stage2")
    
    args = parser.parse_args()
    set_gpu_devices(args.gpu)
    set_seed(999)
    # set_gpu_devices(args.gpu)
    
    # writer = SummaryWriter('./log/tensorboard')
    logger, sign =logger(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets with CausalVidQA paths
    default_stage_subset = ""
    if args.stage_mode in ["stage1_chain", "stage2_qa"] and not args.qtype_subset:
        default_stage_subset = "explanatory,predictive,predictive_reason"
    qtype_subset = args.qtype_subset if args.qtype_subset else default_stage_subset

    dataset_stage_mode = "stage1_chain" if args.stage_mode == "stage1_chain" else "qa"

    train_dataset = VideoQADataset(
        split='train', 
        n_query=args.n_query, 
        obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        split_dir=args.sample_list_path,
        video_feature_path=args.video_feature_path,
        text_annotation_path=args.text_annotation_path,
        qtype=args.qtype,
        qtype_subset=qtype_subset,
        stage_mode=dataset_stage_mode,
        chain_data_root=args.chain_data_root,
        grounding_dino_path=args.grounding_dino_path,
        max_samples=args.max_samples
    )
    val_dataset = VideoQADataset(
        split='val', 
        n_query=args.n_query, 
        obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        split_dir=args.sample_list_path,
        video_feature_path=args.video_feature_path,
        text_annotation_path=args.text_annotation_path,
        qtype=args.qtype,
        qtype_subset=qtype_subset,
        stage_mode=dataset_stage_mode,
        chain_data_root=args.chain_data_root,
        grounding_dino_path=args.grounding_dino_path,
        max_samples=args.max_samples
    )
    test_dataset = VideoQADataset(
        split='test', 
        n_query=args.n_query, 
        obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        split_dir=args.sample_list_path,
        video_feature_path=args.video_feature_path,
        text_annotation_path=args.text_annotation_path,
        qtype=args.qtype,
        qtype_subset=qtype_subset,
        stage_mode=dataset_stage_mode,
        chain_data_root=args.chain_data_root,
        grounding_dino_path=args.grounding_dino_path,
        max_samples=args.max_samples
    )
    
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.bs,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.bs,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.bs,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    
    # hyper setting
    epoch_num = args.epoch
    args.device = device
    config = {**vars(args)}
    if args.grounding_dino_path:
        config['use_grounding_dino'] = True
    model = VideoQAmodel(**config)

    # scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[int(item) for item in args.mile_stone.split(',')], gamma=args.gamma, verbose=True)
    model.to(device)

    if args.stage_mode == "stage2_qa" and args.stage1_checkpoint:
        ckpt = torch.load(args.stage1_checkpoint, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt_state = ckpt['model_state_dict']
        else:
            ckpt_state = ckpt
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        logger.debug(f"Loaded stage1 checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    _apply_stage_freeze(model, args, epoch=1)
    optimizer = _build_optimizer(model, args)
    scheduler_mode = 'min' if args.stage_mode == 'stage1_chain' else 'max'
    scheduler = ReduceLROnPlateau(optimizer, scheduler_mode, factor=args.gamma, patience=args.patience, verbose=True)

    xe = nn.CrossEntropyLoss().to(device)
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    # train & val
    if args.stage_mode == "stage1_chain":
        best_loss = math.inf
        best_epoch = 1
        stage1_save_path = args.save_stage1_checkpoint if args.save_stage1_checkpoint else f'./models/stage1-{sign}.ckpt'

        for epoch in range(1, epoch_num + 1):
            train_loss = train_stage1(model, optimizer, train_loader, device, args, use_amp=args.use_amp, scaler=scaler)
            scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                    'best_loss': best_loss,
                    'epoch': epoch,
                }, stage1_save_path)
            logger.debug("==>Stage1 Epoch:[{}/{}][LR{}][Train Loss: {:.4f}]".format(
                epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss))

        logger.debug("Stage1 Best epoch {} loss {:.4f}; checkpoint {}".format(best_epoch, best_loss, stage1_save_path))
    else:
        best_eval_score = 0.0
        best_epoch = 1
        best_model_path = './models/best_model-{}.ckpt'.format(sign)

        for epoch in range(1, epoch_num + 1):
            if args.stage_mode == "stage2_qa" and args.stage2_freeze_backbone_epochs > 0 and epoch == args.stage2_freeze_backbone_epochs + 1:
                _apply_stage_freeze(model, args, epoch=epoch)
                optimizer = _build_optimizer(model, args)
                scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.gamma, patience=args.patience, verbose=True)
                logger.debug(f"Stage2 unfreeze at epoch {epoch}; rebuilt optimizer")

            train_loss, train_acc = train(model, optimizer, train_loader, xe, device, use_amp=args.use_amp, scaler=scaler)
            eval_score = eval(model, val_loader, device)
            scheduler.step(eval_score)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            test_score = eval(model, test_loader, device)
            logger.debug("==>Epoch:[{}/{}][LR{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f} Test: {:.2f}".
                format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss, train_acc, eval_score, test_score))

        logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

        # predict with best model
        model.load_state_dict(torch.load(best_model_path))
        results, test_acc = predict(model, test_loader, device)
        logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

        result_path = './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
        save_file(results, result_path)
        eval_mc.accuracy_metric_cvid('./prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score))
