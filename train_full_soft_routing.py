#!/usr/bin/env python3
"""Local command-line trainer for TranSTR soft uncertainty routing.

Uses an existing causalvid checkout and local data. It never clones a repository
and never connects to W&B.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train TranSTR with soft uncertainty routing.")
    parser.add_argument("--dinov3-feature-path", required=True,
                        help="Directory containing merged DINOv3 .pt features.")
    parser.add_argument("--gdino-feature-path", required=True,
                        help="Directory containing merged GDINO/FRCNN .pkl features.")
    parser.add_argument("--annotation-path", required=True,
                        help="Directory containing the QA annotations.")
    parser.add_argument("--split-dir", required=True,
                        help="Directory containing train.pkl, valid/val.pkl, and test.pkl.")
    parser.add_argument("--knowledge-bank-path", default=None,
                        help="Optional local causal knowledge-bank JSON.")
    parser.add_argument("--model-dir", default=None,
                        help="Output directory. Default: ./models/<run-name>.")
    parser.add_argument("--run-name", default=None,
                        help="Output/checkpoint name. Default derives from --run-profile.")
    parser.add_argument("--run-profile", default="run1",
                        choices=["baseline", "run1", "run2", "run3"])
    parser.add_argument("--run3-reg-mode", default="dropout", choices=["dropout", "decay"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="Visible GPU index; use -1 with --allow-cpu for CPU.")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional sample cap for a smoke test.")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--skip-train", action="store_true")
    return parser


def main():
    cli_args = build_arg_parser().parse_args()
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    if not (SCRIPT_DIR / "DataLoader.py").exists() or not (SCRIPT_DIR / "networks" / "model.py").exists():
        raise FileNotFoundError(f"Project files are missing under {SCRIPT_DIR}.")
    if cli_args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    elif cli_args.allow_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        raise ValueError("--gpu -1 requires --allow-cpu")
    print(f"Project directory: {SCRIPT_DIR}")

    # Dependency validation
    print('=== Server dependency validation ===')
    import importlib
    import os
    import sys

    # Do not mutate the server environment every time training starts. Check imports
    # first and provide one reproducible repair command if the environment is broken.
    _required_modules = [
        'torch', 'numpy', 'pandas', 'tqdm', 'transformers',
        'huggingface_hub', 'peft', 'sentencepiece', 'einops', 'packaging',
    ]
    _package_errors = []
    for _module_name in _required_modules:
        try:
            importlib.import_module(_module_name)
        except Exception as _exc:
            _package_errors.append(f'{_module_name}: {_exc}')

    if _package_errors:
        _repair_command = (
            f'{sys.executable} -m pip install --upgrade '
            '"huggingface_hub<1.0" "transformers>=4.41,<5.0" '
            'peft sentencepiece einops packaging numpy pandas tqdm'
        )
        raise RuntimeError(
            'Package check failed:\n- '
            + '\n- '.join(_package_errors)
            + f'\n\nRepair the Python environment with:\n{_repair_command}\n'
            + 'Install the CUDA-compatible PyTorch build separately if torch is missing.'
        )

    import huggingface_hub, transformers
    from packaging.version import Version

    _version_errors = []
    if Version(transformers.__version__) >= Version('5.0'):
        _version_errors.append(f'transformers=={transformers.__version__}; required <5.0')
    if Version(huggingface_hub.__version__) >= Version('1.0'):
        _version_errors.append(f'huggingface_hub=={huggingface_hub.__version__}; required <1.0')
    if _version_errors:
        raise RuntimeError(
            'Incompatible package versions:\n- '
            + '\n- '.join(_version_errors)
            + f'\n\nRepair the Python environment with:\n{_repair_command}'
        )
    print(f'huggingface_hub=={huggingface_hub.__version__} | transformers=={transformers.__version__}')

    # Local input validation
    print("=== Local input paths ===")


    def require_local_path(label, raw_path, extension=None):
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"{label} must be a directory: {resolved}")
        if extension and not any(resolved.glob(f"*{extension}")):
            raise FileNotFoundError(f"{label} contains no {extension} files: {resolved}")
        print(f"OK {label}: {resolved}")
        return str(resolved)


    CLIP_FEATURE_PATH = require_local_path("DINOv3 features", cli_args.dinov3_feature_path, ".pt")
    GDINO_FEATURE_PATH = require_local_path("GDINO/FRCNN features", cli_args.gdino_feature_path, ".pkl")
    ANNOTATION_PATH = require_local_path("QA annotations", cli_args.annotation_path)
    SPLIT_DIR = require_local_path("data splits", cli_args.split_dir, ".pkl")

    # Core soft-routing and training functions
    print('=== Imports + Functions (Soft Router + NCOD + HUM) ===')

    import json
    import math
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm.auto import tqdm

    from utils.util import set_seed, set_gpu_devices
    from DataLoader import VideoQADataset
    from networks.model import VideoQAmodel

    def _unpack_batch(batch):
        if len(batch) == 7:
            ff, of, q, a, ans_id, qns_key, q_family_id = batch
        elif len(batch) == 6:
            ff, of, q, a, ans_id, qns_key = batch
            q_family_id = None
        else:
            raise ValueError(f'Unexpected batch format with {len(batch)} elements')
        return ff, of, q, a, ans_id, qns_key, q_family_id

    def _compute_sample_indices(qns_keys, key_to_idx, device):
        idxs = [key_to_idx.get(str(k), -1) for k in qns_keys]
        if any(i < 0 for i in idxs):
            missing = [str(qns_keys[i]) for i, v in enumerate(idxs) if v < 0][:5]
            raise KeyError(f'Missing qns_key in key_to_idx mapping: {missing}')
        return torch.tensor(idxs, dtype=torch.long, device=device)

    _QTYPE_SUFFIXES = [
        'counterfactual_reason',
        'predictive_reason',
        'counterfactual',
        'predictive',
        'explanatory',
        'descriptive',
    ]

    def _split_qns_key(qns_key):
        key = str(qns_key)
        for qtype in _QTYPE_SUFFIXES:
            suffix = f'_{qtype}'
            if key.endswith(suffix):
                return key[:-len(suffix)], qtype
        return key, 'unknown'

    def _compute_acc_all_metrics(type_results):
        mapping = [
            ('Description', 'descriptive'),
            ('Explanation', 'explanatory'),
            ('Predictive-Answer', 'predictive'),
            ('Predictive-Reason', 'predictive_reason'),
            ('Counterfactual-Answer', 'counterfactual'),
            ('Counterfactual-Reason', 'counterfactual_reason'),
        ]
        metrics = {}
        for name, qtype in mapping:
            rows = type_results.get(qtype, [])
            metrics[name] = (sum(1 for r in rows if r['correct']) / len(rows) * 100) if rows else 0.0

        def _hard_metric(type_ans, type_reason):
            ans_by_vid = {r['video_id']: r['correct'] for r in type_results.get(type_ans, [])}
            reason_by_vid = {r['video_id']: r['correct'] for r in type_results.get(type_reason, [])}
            common_vids = set(ans_by_vid) & set(reason_by_vid)
            if not common_vids:
                return 0.0
            both_correct = sum(1 for vid in common_vids if ans_by_vid[vid] and reason_by_vid[vid])
            return both_correct / len(common_vids) * 100

        metrics['PAR'] = _hard_metric('predictive', 'predictive_reason')
        metrics['CAR'] = _hard_metric('counterfactual', 'counterfactual_reason')
        metrics['Acc_ALL'] = (metrics['Description'] + metrics['Explanation'] + metrics['PAR'] + metrics['CAR']) / 4.0
        return metrics

    def _hard_negative_similarity(cand_feat, target):
        """Maximum non-gold/gold cosine similarity r_j in [0, 1]."""
        if cand_feat is None:
            return torch.zeros_like(target, dtype=torch.float32)
        with torch.no_grad():
            cand_norm = F.normalize(cand_feat.detach(), dim=-1)
            gold = cand_norm.gather(1, target.view(-1, 1, 1).expand(-1, 1, cand_norm.size(-1))).squeeze(1)
            sims = torch.bmm(cand_norm, gold.unsqueeze(-1)).squeeze(-1)
            sims.scatter_(1, target.view(-1, 1), -1.0)
            return sims.max(dim=1).values.clamp(min=0.0, max=1.0)

    def _hard_negative_weights(similarity, enabled=False, max_weight=1.5):
        if (not enabled) or max_weight <= 1.0:
            return torch.ones_like(similarity)
        return 1.0 + (float(max_weight) - 1.0) * similarity

    class SoftUncertaintyRouter(nn.Module):
        """Lightweight sample-wise gate: 0 -> LUM, 1 -> HUM."""
        def __init__(self, d_model, hidden_dim=128, dropout=0.1):
            super().__init__()
            self.h_norm = nn.LayerNorm(d_model)
            self.signal_norm = nn.LayerNorm(5)
            self.net = nn.Sequential(
                nn.Linear(d_model + 5, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, h, uncertainty, ce, entropy, margin, hard_negative_similarity):
            # Detach dynamic signals to avoid a shortcut through CE/entropy features.
            h = self.h_norm(h.detach())
            signals = torch.stack([
                uncertainty,
                torch.log1p(ce.clamp(0.0, 20.0)),
                entropy,
                margin,
                hard_negative_similarity,
            ], dim=1)
            z = torch.cat([h, self.signal_norm(signals.detach())], dim=1)
            return torch.sigmoid(self.net(z)).squeeze(-1)

    def _update_ema_model(ema_model, model, decay):
        if ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
            for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

    def train_epoch_integrated(
        model, router, optimizer_model, optimizer_u, U, loader, xe, bce, device, epoch, key_to_idx,
        accumulation_steps=4, hum_alpha=1.0, lambda_verifier=0.2, lambda_knowledge=0.1,
        lambda_prior=0.1, use_hard_neg_mining=False, hard_neg_weight_max=1.5,
        ema_model=None, ema_decay=0.999,
        scheduler=None, scheduler_step_per_batch=False
    ):
        model.train()
        router.train()
        total_loss, total_l1, total_l2 = 0.0, 0.0, 0.0
        total_verifier, total_knowledge, total_prior = 0.0, 0.0, 0.0
        gate_sum = gate_de_sum = gate_pc_sum = 0.0
        gate_count = gate_de_count = gate_pc_count = 0
        correct, total = 0, 0
        optimizer_model.zero_grad()
        optimizer_u.zero_grad()

        pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, batch in enumerate(pbar):
            ff, of, q, a, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
            ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)

            if q_family_id is None:
                q_family_id = torch.zeros_like(tgt)
            else:
                q_family_id = q_family_id.to(device)

            sample_indices = _compute_sample_indices(qns_keys, key_to_idx, device)
            out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
            logits = out['logits']
            fused_logits = out.get('fused_score', logits)
            verifier_logits = out.get('verifier_logits', logits)
            knowledge_logits = out.get('knowledge_score', None)
            cand_feat = out.get('cand_feat', None)

            probs = torch.softmax(logits, dim=1)
            y_onehot = F.one_hot(tgt, num_classes=logits.size(-1)).float()
            u_batch = U[sample_indices].unsqueeze(1)

            ce_per_sample = -torch.sum(y_onehot * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
            shifted_probs = torch.clamp(probs + (u_batch.detach() * y_onehot), min=1e-12, max=1.0)
            lum_loss = -torch.sum(y_onehot * torch.log(shifted_probs), dim=1)
            hum_loss = (1.0 + hum_alpha * u_batch.detach().squeeze(1)) * ce_per_sample

            # z_j = [h_j, U_j, CE_j, H(p_j), Delta_j, r_j]
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)
            entropy = entropy / math.log(max(2, logits.size(1)))
            top2 = probs.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            hard_neg_similarity = _hard_negative_similarity(cand_feat, tgt)
            gate = router(
                out['mem_pool'],
                u_batch.detach().squeeze(1),
                ce_per_sample.detach(),
                entropy.detach(),
                margin.detach(),
                hard_neg_similarity,
            )
            family_prior = (q_family_id >= 2).float()
            prior_loss = F.binary_cross_entropy(gate.clamp(1e-6, 1.0 - 1e-6), family_prior)
            routed_loss = (1.0 - gate) * lum_loss + gate * hum_loss
            hard_neg_w = _hard_negative_weights(
                hard_neg_similarity, enabled=use_hard_neg_mining, max_weight=hard_neg_weight_max
            )
            l1 = (routed_loss * hard_neg_w).mean()

            verifier_loss = bce(verifier_logits, y_onehot)
            if knowledge_logits is not None:
                knowledge_loss = bce(knowledge_logits, y_onehot)
            else:
                knowledge_loss = torch.tensor(0.0, device=device)

            model_loss = l1 + lambda_prior * prior_loss + lambda_verifier * verifier_loss + lambda_knowledge * knowledge_loss
            (model_loss / accumulation_steps).backward()

            shifted_det = probs.detach() + (u_batch * y_onehot)
            l2 = F.mse_loss(shifted_det, y_onehot)
            (l2 / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(router.parameters()), max_norm=1.0)
                optimizer_model.step()
                _update_ema_model(ema_model, model, ema_decay)
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()
                optimizer_model.zero_grad()
                optimizer_u.step()
                optimizer_u.zero_grad()
                with torch.no_grad():
                    U.clamp_(0.0, 0.99)

            total_l1 += l1.item()
            total_l2 += l2.item()
            total_verifier += verifier_loss.item()
            total_knowledge += knowledge_loss.item()
            total_prior += prior_loss.item()
            total_loss += (model_loss + l2).item()
            with torch.no_grad():
                de_mask = family_prior < 0.5
                pc_mask = ~de_mask
                gate_sum += gate.sum().item()
                gate_count += gate.numel()
                if de_mask.any():
                    gate_de_sum += gate[de_mask].sum().item()
                    gate_de_count += int(de_mask.sum().item())
                if pc_mask.any():
                    gate_pc_sum += gate[pc_mask].sum().item()
                    gate_pc_count += int(pc_mask.sum().item())
            correct += (fused_logits.argmax(-1) == tgt).sum().item()
            total += tgt.size(0)

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'route': total_l1 / (batch_idx + 1),
                'prior': total_prior / (batch_idx + 1),
                'gate': gate_sum / max(gate_count, 1),
                'acc': correct / max(total, 1) * 100
            })

        n = len(loader)
        return (
            total_loss / n,
            total_l1 / n,
            total_l2 / n,
            total_verifier / n,
            total_knowledge / n,
            total_prior / n,
            gate_sum / max(gate_count, 1),
            gate_de_sum / max(gate_de_count, 1),
            gate_pc_sum / max(gate_pc_count, 1),
            correct / max(total, 1) * 100
        )

    def eval_epoch(model, loader, device):
        model.eval()
        correct, total = 0, 0
        type_results = {}
        with torch.no_grad():
            for batch in loader:
                ff, of, q, a, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
                ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)
                q_family_id = q_family_id.to(device) if q_family_id is not None else None
                out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
                logits = out.get('fused_score', out['logits'])
                preds = logits.argmax(-1)
                correct += (preds == tgt).sum().item()
                total += tgt.size(0)

                for key, pred, target in zip(qns_keys, preds.detach().cpu().tolist(), tgt.detach().cpu().tolist()):
                    video_id, qtype = _split_qns_key(key)
                    type_results.setdefault(qtype, []).append({
                        'video_id': video_id,
                        'correct': bool(int(pred) == int(target)),
                    })

        metrics = _compute_acc_all_metrics(type_results)
        metrics['Plain_Acc'] = correct / max(total, 1) * 100
        return metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print('Imports and functions defined for integrated training.')

    # Local run configuration
    print("=== Local run configuration ===")

    # ============================================
    # 3-RUN TUNING PRESETS
    # ============================================
    RUN_TRAINING = not cli_args.skip_train
    RUN_PROFILE = cli_args.run_profile
    RUN_VARIANT = 'weak_soft_router_lora_hn_ema_cos'
    RUN3_REG_MODE = cli_args.run3_reg_mode

    RUN_PROFILES = {
        'baseline': {
            'epoch': 10, 'lr': 1e-5,
            'lambda_verifier': 0.3, 'lambda_knowledge': 0.2,
            'early_stop_start_epoch': 5, 'early_stop_patience': 4,
            'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
        },
        'run1': {
            'epoch': 10, 'lr': 1e-5,
            'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
            'early_stop_start_epoch': 5, 'early_stop_patience': 4,
            'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
        },
        'run2': {
            'epoch': 10, 'lr': 8e-6,
            'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
            'early_stop_start_epoch': 6, 'early_stop_patience': 5,
            'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
        },
        'run3': {
            'epoch': 10, 'lr': 8e-6,
            'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
            'early_stop_start_epoch': 6, 'early_stop_patience': 5,
            'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
        },
    }

    if RUN_PROFILE not in RUN_PROFILES:
        raise ValueError(f'Invalid RUN_PROFILE={RUN_PROFILE}')

    RUN_TAG = cli_args.run_name or f'{RUN_PROFILE}_{RUN_VARIANT}'
    MODEL_DIR = str(Path(cli_args.model_dir).expanduser().resolve()) if cli_args.model_dir else str((Path.cwd() / 'models' / RUN_TAG).resolve())
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_FILENAME = f'best_model_gdinofrcnn_soft_router_{RUN_TAG}.ckpt'
    LATEST_CKPT_FILENAME = f'latest_checkpoint_gdinofrcnn_soft_router_{RUN_TAG}.ckpt'
    TRAIN_HISTORY_FILENAME = f'train_history_gdinofrcnn_soft_router_{RUN_TAG}.csv'
    PREDICTIONS_CSV_FILENAME = f'test_predictions_gdinofrcnn_soft_router_{RUN_TAG}.csv'
    METRICS_JSON_FILENAME = f'final_metrics_gdinofrcnn_soft_router_{RUN_TAG}.json'
            
    FEAT_DIM = 1024  # DINOv3 frame
    OBJ_DIM  = 2820  # FRCNN(2048) + DeBERTa cls(768) + bbox(4)
    OBJ_BBOX_DIM = 4
    print(f'\nBackbone: DINOv3 ({FEAT_DIM}-d frame) + GroundingDINO+FRCNN ({OBJ_DIM}-d obj, bbox split={OBJ_BBOX_DIM})')
    print(f'Run profile: {RUN_PROFILE}')

    class Config:
        # Paths
        video_feature_root = CLIP_FEATURE_PATH
        grounding_dino_path = GDINO_FEATURE_PATH
        sample_list_path = ANNOTATION_PATH
        split_dir_txt = SPLIT_DIR

        # Model architecture
        topK_frame = 16
        objs = 12
        frames = 16
        select_frames = 5
        topK_obj = 12

        frame_feat_dim = FEAT_DIM
        obj_feat_dim = OBJ_DIM
        use_grounding_dino = True

        # GDINO object encoding fixes
        obj_use_bbox_pos_embed = True
        obj_bbox_dim = OBJ_BBOX_DIM
        obj_hard_gather_from_frame = True

        d_model = 768
        word_dim = 768
        nheads = 8
        num_encoder_layers = 2
        num_decoder_layers = 2
        normalize_before = True
        activation = 'gelu'
        dropout = 0.3
        encoder_dropout = 0.3

        # Text encoder
        text_encoder_type = 'microsoft/deberta-base'
        freeze_text_encoder = False
        text_encoder_lr = 5e-6
        text_pool_mode = 1
        use_lora = True
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.1
        lora_target_modules = ['query_proj', 'key_proj', 'value_proj']

        # Training
        bs = 8
        accumulation_steps = 4
        lr = 1e-5
        epoch = 10
        gpu = 0
        gamma = 0.5
        decay = 1e-4
        n_query = 5
        lambda_verifier = 0.3
        lambda_knowledge = 0.2
        return_family_id = True

        # LR scheduler + early stopping
        lr_schedule = 'cosine_warmup'
        warmup_epochs = 1
        lr_patience = 1
        min_lr = 1e-7
        early_stop_patience = 4
        early_stop_min_delta = 0.05
        early_stop_start_epoch = 5

        # Generic training improvements
        use_hard_neg_mining = True
        hard_neg_weight_max = 1.5
        use_ema = True
        ema_decay = 0.999

        # NCOD + learned soft LUM/HUM routing
        ncod_u_lr = 0.1
        hum_alpha = 1.0
        router_hidden_dim = 128
        router_dropout = 0.1
        router_lr = 1e-4
        lambda_prior = 0.1

        # Aux warmup
        aux_warmup_epochs = 2

        # Other
        hard_eval = False
        pos_ratio = 1.0
        neg_ratio = 1.0
        a = 1.0
        num_workers = 4

    for _k, _v in RUN_PROFILES[RUN_PROFILE].items():
        setattr(Config, _k, _v)

    if RUN_PROFILE == 'run3':
        if RUN3_REG_MODE == 'dropout':
            Config.dropout = 0.25
            Config.encoder_dropout = 0.25
        elif RUN3_REG_MODE == 'decay':
            Config.decay = 8e-5
        else:
            raise ValueError(f'Invalid RUN3_REG_MODE={RUN3_REG_MODE}')

    # Explicit command-line values override the selected profile.
    Config.gpu = cli_args.gpu
    Config.num_workers = cli_args.num_workers
    Config.max_samples = cli_args.max_samples
    if cli_args.bs is not None:
        Config.bs = cli_args.bs
    if cli_args.accumulation_steps is not None:
        Config.accumulation_steps = cli_args.accumulation_steps
    if cli_args.epochs is not None:
        Config.epoch = cli_args.epochs

    args = Config()

    if args.text_encoder_type != 'microsoft/deberta-base':
        raise ValueError('Train notebook uses DeBERTa v1 only.')

    set_gpu_devices(args.gpu)
    set_seed(cli_args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not cli_args.allow_cpu:
        raise RuntimeError(
            'CUDA is required. Verify the server PyTorch CUDA build, '
            'or pass --allow-cpu only for debugging.'
        )
    print(f'\nDevice: {device}')
    print(f'Run tag: {RUN_TAG}')
    print(f'Config: frame_feat_dim={args.frame_feat_dim}, obj_feat_dim={args.obj_feat_dim}, objs={args.objs}, select_frames={args.select_frames}')
    print(f'use_grounding_dino={args.use_grounding_dino} -> obj_sorter SKIPPED')
    print(f'obj_use_bbox_pos_embed={args.obj_use_bbox_pos_embed} | obj_hard_gather_from_frame={args.obj_hard_gather_from_frame}')
    print(f'Effective bs: physical={args.bs} x accum={args.accumulation_steps} = {args.bs * args.accumulation_steps}')
    print(f'lr={args.lr} | text_encoder_lr={args.text_encoder_lr} | decay={args.decay}')
    print(f'use_lora={args.use_lora} | hard_neg={args.use_hard_neg_mining} (max={args.hard_neg_weight_max}) | ema={args.use_ema} (decay={args.ema_decay})')
    print(f'lr_schedule={args.lr_schedule} | warmup_epochs={args.warmup_epochs}')
    print(f'aux_warmup_epochs={args.aux_warmup_epochs} | verifier={args.lambda_verifier} | knowledge={args.lambda_knowledge}')
    print(f'soft_router: hidden={args.router_hidden_dim} | lr={args.router_lr} | lambda_prior={args.lambda_prior}')
    print(f'Early stop: start={args.early_stop_start_epoch}, patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}')
    print(f'Model file: {MODEL_FILENAME}')

    # Create Datasets
    print('=== Datasets ===')

    train_ds = VideoQADataset(
        split='train', n_query=args.n_query, obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_root,
        grounding_dino_path=args.grounding_dino_path,
        split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
        max_samples=args.max_samples, verbose=True, return_family_id=args.return_family_id
    )
    val_ds = VideoQADataset(
        split='val', n_query=args.n_query, obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_root,
        grounding_dino_path=args.grounding_dino_path,
        split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
        max_samples=args.max_samples, verbose=True, return_family_id=args.return_family_id
    )
    test_ds = VideoQADataset(
        split='test', n_query=args.n_query, obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_root,
        grounding_dino_path=args.grounding_dino_path,
        split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
        max_samples=args.max_samples, verbose=True, return_family_id=args.return_family_id
    )

    train_loader = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_sample_keys = [f"{row.video_id}_{row.type}" for row in train_ds.sample_list.itertuples(index=False)]
    train_key_to_idx = {k: i for i, k in enumerate(train_sample_keys)}

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    # Model + Optimizers + NCOD U + Generic Improvements
    print('=== Model ===')
    cfg = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
    cfg['device'] = device
    cfg['topK_frame'] = args.select_frames
    model = VideoQAmodel(**cfg)
    model.to(device)
    router = SoftUncertaintyRouter(
        d_model=args.d_model,
        hidden_dim=args.router_hidden_dim,
        dropout=args.router_dropout,
    ).to(device)

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    non_text_params = []
    text_base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'text_encoder' in name:
            text_base_params.append(p)
        else:
            non_text_params.append(p)

    param_groups = []
    if len(non_text_params) > 0:
        param_groups.append({'params': non_text_params, 'lr': args.lr, 'weight_decay': args.decay})
    if len(text_base_params) > 0:
        param_groups.append({'params': text_base_params, 'lr': args.text_encoder_lr, 'weight_decay': args.decay})
    param_groups.append({'params': router.parameters(), 'lr': args.router_lr, 'weight_decay': args.decay})
    if len(param_groups) == 0:
        raise RuntimeError('No trainable parameters found for optimizer_model.')

    optimizer_model = torch.optim.AdamW(param_groups)

    if args.lr_schedule == 'cosine_warmup':
        steps_per_epoch = max(1, math.ceil(len(train_loader) / args.accumulation_steps))
        total_steps = max(1, args.epoch * steps_per_epoch)
        warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

        def _lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            min_ratio = args.min_lr / max(args.lr, 1e-12)
            return max(min_ratio, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=_lr_lambda)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer_model,
            mode='max',
            factor=args.gamma,
            patience=args.lr_patience,
            threshold=args.early_stop_min_delta,
            threshold_mode='abs',
            min_lr=args.min_lr
        )

    U = torch.nn.Parameter(torch.full((len(train_ds),), 1e-8, dtype=torch.float32, device=device))
    optimizer_u = torch.optim.SGD([U], lr=args.ncod_u_lr)

    xe = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    save_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M')
    print(f'Router params: {sum(p.numel() for p in router.parameters())/1e3:.1f}K')
    print(f'Text-encoder trainable params: {sum(p.numel() for p in text_base_params)/1e6:.3f}M')
    print(f'EMA enabled: {ema_model is not None}')
    print(f'Scheduler: {args.lr_schedule}')
    print(f'U shape: {tuple(U.shape)}')

    # Init W&B + Resume Checkpoint
    print('=== Initialize W&B Run ===')

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    history_rows = []

    LATEST_CKPT_PATH = os.path.join(MODEL_DIR, LATEST_CKPT_FILENAME)
    TRAIN_HISTORY_CSV_PATH = os.path.join(MODEL_DIR, TRAIN_HISTORY_FILENAME)

    RESUME_FROM_CHECKPOINT = cli_args.resume
    LOCAL_RESUME_PATH = cli_args.resume_path or LATEST_CKPT_PATH

    print('Tracking: local checkpoints, CSV, and JSON only')

    def _load_resume_checkpoint(path, map_location):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return torch.load(path, map_location=map_location)

    if RESUME_FROM_CHECKPOINT:
        print('Resume enabled from local checkpoint')
        try:
            checkpoint = None
            resume_path = None

            resume_path = LOCAL_RESUME_PATH
            checkpoint = _load_resume_checkpoint(resume_path, device)

            ckpt_state = checkpoint['model_state_dict']
            model_state = model.state_dict()
            filtered_state = {}
            skipped_keys = []
            for k, v in ckpt_state.items():
                if k in model_state and v.shape != model_state[k].shape:
                    skipped_keys.append(f'{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}')
                else:
                    filtered_state[k] = v

            if skipped_keys:
                print(f'Skipped {len(skipped_keys)} keys due to shape mismatch')

            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            if missing:
                print(f'Warning: missing model keys when resume: {len(missing)}')
            if unexpected:
                print(f'Warning: unexpected model keys when resume: {len(unexpected)}')

            router_restored = checkpoint.get('router_state_dict') is not None
            if router_restored:
                router.load_state_dict(checkpoint['router_state_dict'])
            else:
                print('Warning: checkpoint has no router state; router starts fresh.')

            if not skipped_keys and router_restored:
                if 'optimizer_model_state_dict' in checkpoint:
                    optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
                if 'optimizer_u_state_dict' in checkpoint:
                    optimizer_u.load_state_dict(checkpoint['optimizer_u_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if ema_model is not None and checkpoint.get('ema_model_state_dict') is not None:
                    ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
                if 'U' in checkpoint:
                    with torch.no_grad():
                        u_ckpt = checkpoint['U'].to(device).float().view(-1)
                        n = min(u_ckpt.numel(), U.numel())
                        U[:n].copy_(u_ckpt[:n])

                best_acc = float(checkpoint.get('best_acc', 0.0))
                start_epoch = int(checkpoint.get('epoch', 0)) + 1
                best_epoch = int(checkpoint.get('best_epoch', 0))
                epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', 0))
                history = checkpoint.get('history', history)
                history_rows = checkpoint.get('history_rows', history_rows)

                if os.path.exists(TRAIN_HISTORY_CSV_PATH):
                    try:
                        history_rows = pd.read_csv(TRAIN_HISTORY_CSV_PATH).to_dict('records')
                        print(f'Loaded history CSV with {len(history_rows)} rows')
                    except Exception as csv_err:
                        print(f'Warning: failed to load history CSV: {csv_err}')
            else:
                print('Optimizer/scheduler/U NOT restored because model/router state is incompatible.')

            print(f'Resumed from: {resume_path}')
            print(f'Start epoch: {start_epoch} | Best acc: {best_acc:.2f}% | Best epoch: {best_epoch}')
        except Exception as e:
            print(f'Warning: resume failed, starting from scratch. Error: {e}')
    else:
        print('Resume disabled. Training starts from epoch 1.')

    # Integrated Training Loop + Checkpoint/CSV Logging + Early Stopping
    print('=== Training ===')

    if RUN_TRAINING:
        stop_training = False
        for ep in range(start_epoch, args.epoch + 1):
            print(f'\nEpoch {ep}/{args.epoch}')

            # Aux warmup: keep verifier/knowledge losses off in early epochs
            if ep <= args.aux_warmup_epochs:
                eff_lambda_verifier = 0.0
                eff_lambda_knowledge = 0.0
            else:
                eff_lambda_verifier = args.lambda_verifier
                eff_lambda_knowledge = args.lambda_knowledge

            (total_loss, l1, l2, verifier_loss, knowledge_loss, prior_loss,
             gate_mean, gate_de_prior_mean, gate_pc_prior_mean, train_acc) = train_epoch_integrated(
                model=model,
                router=router,
                optimizer_model=optimizer_model,
                optimizer_u=optimizer_u,
                U=U,
                loader=train_loader,
                xe=xe,
                bce=bce,
                device=device,
                epoch=ep,
                key_to_idx=train_key_to_idx,
                accumulation_steps=args.accumulation_steps,
                hum_alpha=args.hum_alpha,
                lambda_verifier=eff_lambda_verifier,
                lambda_knowledge=eff_lambda_knowledge,
                lambda_prior=args.lambda_prior,
                use_hard_neg_mining=args.use_hard_neg_mining,
                hard_neg_weight_max=args.hard_neg_weight_max,
                ema_model=ema_model,
                ema_decay=args.ema_decay,
                scheduler=scheduler,
                scheduler_step_per_batch=(args.lr_schedule == 'cosine_warmup')
            )

            eval_model = ema_model if ema_model is not None else model
            val_metrics = eval_epoch(eval_model, val_loader, device)
            val_acc = float(val_metrics['Acc_ALL'])
            val_plain_acc = float(val_metrics['Plain_Acc'])
            if args.lr_schedule != 'cosine_warmup':
                scheduler.step(val_acc)

            history['train_loss'].append(total_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            current_lrs = [pg['lr'] for pg in optimizer_model.param_groups]
            min_lr_now = float(min(current_lrs))
            max_lr_now = float(max(current_lrs))

            improved = val_acc > (best_acc + args.early_stop_min_delta)
            if improved:
                best_acc = val_acc
                best_epoch = ep
                epochs_without_improvement = 0
                print(f'New best val_acc(Acc_ALL)={best_acc:.2f}% at epoch {best_epoch} | val_plain_acc={val_plain_acc:.2f}%')
            elif ep >= args.early_stop_start_epoch:
                epochs_without_improvement += 1
                print(
                    f'No significant improvement for {epochs_without_improvement} epoch(s) '
                    f'(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})'
                )

            epoch_row = {
                'epoch': ep,
                'train_total_loss': float(total_loss),
                'train_l1': float(l1),
                'train_l2': float(l2),
                'train_verifier_loss': float(verifier_loss),
                'train_knowledge_loss': float(knowledge_loss),
                'train_prior_loss': float(prior_loss),
                'router_gate_mean': float(gate_mean),
                'router_gate_de_prior_mean': float(gate_de_prior_mean),
                'router_gate_pc_prior_mean': float(gate_pc_prior_mean),
                'router_gate_family_gap': float(gate_pc_prior_mean - gate_de_prior_mean),
                'train_acc': float(train_acc),
                'val_acc': float(val_acc),
                'val_plain_acc': float(val_plain_acc),
                'val_Description': float(val_metrics.get('Description', 0.0)),
                'val_Explanation': float(val_metrics.get('Explanation', 0.0)),
                'val_PAR': float(val_metrics.get('PAR', 0.0)),
                'val_CAR': float(val_metrics.get('CAR', 0.0)),
                'lambda_verifier_eff': float(eff_lambda_verifier),
                'lambda_knowledge_eff': float(eff_lambda_knowledge),
                'u_mean': float(U.detach().mean().item()),
                'u_max': float(U.detach().max().item()),
                'lr_main_min': min_lr_now,
                'lr_main_max': max_lr_now,
                'best_acc_so_far': float(best_acc),
                'best_epoch_so_far': int(best_epoch),
                'epochs_without_improvement': int(epochs_without_improvement)
            }
            history_rows.append(epoch_row)
            pd.DataFrame(history_rows).to_csv(TRAIN_HISTORY_CSV_PATH, index=False)

            ckpt = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'router_state_dict': router.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'optimizer_u_state_dict': optimizer_u.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'epochs_without_improvement': epochs_without_improvement,
                'history': history,
                'history_rows': history_rows,
                'U': U.detach().cpu(),
                'train_sample_keys': train_sample_keys
            }

            torch.save(ckpt, LATEST_CKPT_PATH)

            if improved:
                torch.save(ckpt, save_path)
            if ep >= args.early_stop_start_epoch and epochs_without_improvement >= args.early_stop_patience:
                print(f'Early stopping at epoch {ep}. Best val_acc(Acc_ALL)={best_acc:.2f}% at epoch {best_epoch}.')
                stop_training = True
                break

        if os.path.exists(save_path):
            best_ckpt = torch.load(save_path, map_location=device)
            best_state = best_ckpt.get('ema_model_state_dict') or best_ckpt['model_state_dict']
            model.load_state_dict(best_state, strict=False)
            if best_ckpt.get('router_state_dict') is not None:
                router.load_state_dict(best_ckpt['router_state_dict'])
            if ema_model is not None and best_ckpt.get('ema_model_state_dict') is not None:
                ema_model.load_state_dict(best_ckpt['ema_model_state_dict'], strict=False)
            print(f'Loaded best checkpoint from epoch {best_epoch} for final evaluation.')

        if not stop_training:
            print(f'Training finished all {args.epoch} epochs. Best Val Acc_ALL: {best_acc:.2f}%')
    else:
        print('Skipping training')

    # Detailed Evaluation + Memory Post-check + CSV export
    print('=== Detailed Evaluation + Memory Post-check ===')
    from networks.knowledge_retriever import CausalKnowledgeRetriever

    CSV_OUTPUT_PATH = os.path.join(MODEL_DIR, PREDICTIONS_CSV_FILENAME)
    COMPARISON_CSV_PATH = os.path.join(MODEL_DIR, 'run_comparison_gdino_3run.csv')

    TOPK_KNOWLEDGE = 5
    MEMORY_PASS_THRESHOLD = 0.15
    MEMORY_GATE_ENABLED = True
    MEMORY_MARGIN = 0.05

    def _resolve_kb_path():
        candidates = []
        if cli_args.knowledge_bank_path:
            candidates.append(Path(cli_args.knowledge_bank_path).expanduser())
        candidates.extend([
            SCRIPT_DIR / 'data' / 'causal_knowledge_bank.json',
            Path.cwd() / 'data' / 'causal_knowledge_bank.json',
        ])
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return str(resolved)
        if cli_args.knowledge_bank_path:
            raise FileNotFoundError(
                f'Knowledge bank not found: {Path(cli_args.knowledge_bank_path).expanduser().resolve()}'
            )
        return None

    KB_PATH = _resolve_kb_path()
    retriever = CausalKnowledgeRetriever(KB_PATH, topk=TOPK_KNOWLEDGE) if KB_PATH else None
    print(f'Knowledge bank path: {KB_PATH if KB_PATH else "NOT FOUND (memory check disabled)"}')

    def _qtype_to_family(qtype):
        qtype = str(qtype)
        valid = {'descriptive', 'explanatory', 'predictive', 'predictive_reason', 'counterfactual', 'counterfactual_reason'}
        return qtype if qtype in valid else 'descriptive'

    def _score_candidate_with_memory(question, candidate, q_family, video_anchors=None):
        if retriever is None:
            return 0.0, []
        hits = retriever.retrieve(
            question=str(question),
            candidate=str(candidate),
            video_anchors=video_anchors or [],
            q_family=str(q_family),
            topk=TOPK_KNOWLEDGE
        )
        top_score = max([float(h.get('score', 0.0)) for h in hits], default=0.0)
        return top_score, hits

    def _build_eval_meta_map(loader):
        dataset = getattr(loader, 'dataset', None)
        sample_list = getattr(dataset, 'sample_list', None) if dataset is not None else None
        meta_map = {}
        if sample_list is None:
            return meta_map
        for _, row in sample_list.iterrows():
            vid = str(row.get('video_id', ''))
            qtype = str(row.get('type', 'unknown'))
            qns_key = f'{vid}_{qtype}'
            meta_map[qns_key] = {
                'video_id': vid,
                'question_type': qtype,
                'question': str(row.get('question', '')),
                'answers': [str(row.get(f'a{i}', '')) for i in range(5)]
            }
        return meta_map

    def evaluate_detailed_v2(model, loader, device):
        model.eval()
        type_results = {}
        prediction_rows = []
        meta_map = _build_eval_meta_map(loader)

        memory_match_flags = []
        memory_pass_flags = []
        memory_gate_correct_flags = []

        with torch.no_grad():
            for batch in tqdm(loader):
                ff, of, qns, ans_word, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
                ff, of = ff.to(device), of.to(device)
                q_family_id = q_family_id.to(device) if q_family_id is not None else None

                out = model(ff, of, qns, ans_word, return_aux=True, q_family_id=q_family_id)
                logits = out.get('fused_score', out['logits'])
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1)
                targets = ans_id.numpy()

                for i, (key, pred, target, prob_vec) in enumerate(zip(qns_keys, preds, targets, probs)):
                    meta = meta_map.get(str(key), {})
                    qtype = str(meta.get('question_type', 'unknown'))
                    q_family = _qtype_to_family(qtype)
                    video_id = str(meta.get('video_id', str(key)))
                    question = str(meta.get('question', qns[i]))
                    answers = meta.get('answers', ['', '', '', '', ''])
                    if len(answers) < 5:
                        answers += [''] * (5 - len(answers))
                    answers = answers[:5]

                    correct_idx = int(target)
                    predicted_idx = int(pred)
                    is_correct = int(correct_idx == predicted_idx)

                    cand_mem_scores = []
                    for cand in answers:
                        score, _ = _score_candidate_with_memory(question, cand, q_family, video_anchors=[video_id])
                        cand_mem_scores.append(float(score))

                    mem_best_idx = int(np.argmax(cand_mem_scores)) if len(cand_mem_scores) > 0 else predicted_idx
                    pred_mem_score = float(cand_mem_scores[predicted_idx]) if len(cand_mem_scores) > predicted_idx else 0.0
                    gt_mem_score = float(cand_mem_scores[correct_idx]) if len(cand_mem_scores) > correct_idx else 0.0

                    memory_match_pred = int(predicted_idx == mem_best_idx)
                    memory_pass_pred = int(pred_mem_score >= MEMORY_PASS_THRESHOLD)

                    gated_idx = predicted_idx
                    if MEMORY_GATE_ENABLED and len(cand_mem_scores) > 0:
                        if (cand_mem_scores[mem_best_idx] - pred_mem_score) >= MEMORY_MARGIN:
                            gated_idx = mem_best_idx
                    gated_correct = int(gated_idx == correct_idx)

                    memory_match_flags.append(memory_match_pred)
                    memory_pass_flags.append(memory_pass_pred)
                    memory_gate_correct_flags.append(gated_correct)

                    type_results.setdefault(qtype, []).append({
                        'video_id': video_id,
                        'pred': predicted_idx,
                        'target': correct_idx,
                        'correct': bool(is_correct)
                    })

                    prediction_rows.append({
                        'video_id': video_id,
                        'question_type': qtype,
                        'question': question,
                        'correct_idx': correct_idx,
                        'predicted_idx': predicted_idx,
                        'predicted_idx_after_memory_gate': int(gated_idx),
                        'is_correct': is_correct,
                        'is_correct_after_memory_gate': gated_correct,
                        'confidence': float(prob_vec[predicted_idx]),
                        'memory_best_idx': mem_best_idx,
                        'memory_match_pred': memory_match_pred,
                        'memory_pass_pred': memory_pass_pred,
                        'memory_pred_score': pred_mem_score,
                        'memory_gt_score': gt_mem_score,
                        'a0': answers[0], 'prob_a0': float(prob_vec[0]),
                        'a1': answers[1], 'prob_a1': float(prob_vec[1]),
                        'a2': answers[2], 'prob_a2': float(prob_vec[2]),
                        'a3': answers[3], 'prob_a3': float(prob_vec[3]),
                        'a4': answers[4], 'prob_a4': float(prob_vec[4]),
                        'predicted_answer': answers[predicted_idx],
                        'correct_answer': answers[correct_idx]
                    })

        prediction_df = pd.DataFrame(prediction_rows)
        prediction_df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f'Saved detailed predictions CSV: {CSV_OUTPUT_PATH}')

        metrics = {}
        mapping = [
            ('Description', 'descriptive'),
            ('Explanation', 'explanatory'),
            ('Predictive-Answer', 'predictive'),
            ('Predictive-Reason', 'predictive_reason'),
            ('Counterfactual-Answer', 'counterfactual'),
            ('Counterfactual-Reason', 'counterfactual_reason')
        ]
        for name, qtype in mapping:
            rows = type_results.get(qtype, [])
            total = len(rows)
            correct = sum(1 for r in rows if r['correct'])
            metrics[name] = (correct / total * 100) if total > 0 else 0.0

        def _calc_hard_metric(type_ans, type_reason):
            if type_ans not in type_results or type_reason not in type_results:
                return 0.0
            ans_by_vid = {r['video_id']: r['correct'] for r in type_results[type_ans]}
            reason_by_vid = {r['video_id']: r['correct'] for r in type_results[type_reason]}
            common_vids = set(ans_by_vid.keys()) & set(reason_by_vid.keys())
            if len(common_vids) == 0:
                return 0.0
            both_correct = sum(1 for vid in common_vids if ans_by_vid[vid] and reason_by_vid[vid])
            return both_correct / len(common_vids) * 100

        metrics['PAR'] = _calc_hard_metric('predictive', 'predictive_reason')
        metrics['CAR'] = _calc_hard_metric('counterfactual', 'counterfactual_reason')
        metrics['Acc_ALL'] = (
            metrics['Description'] + metrics['Explanation'] + metrics['PAR'] + metrics['CAR']
        ) / 4.0

        metrics['Memory_Consistency'] = float(np.mean(memory_match_flags) * 100) if len(memory_match_flags) > 0 else 0.0
        metrics['Memory_Pass_Rate'] = float(np.mean(memory_pass_flags) * 100) if len(memory_pass_flags) > 0 else 0.0
        metrics['Memory_Gated_Acc'] = float(np.mean(memory_gate_correct_flags) * 100) if len(memory_gate_correct_flags) > 0 else 0.0

        metrics['WeightedScore_WeakPriority'] = (
            0.35 * metrics['Predictive-Reason'] +
            0.35 * metrics['Counterfactual-Reason'] +
            0.20 * metrics['Acc_ALL'] +
            0.10 * float(best_acc)
        )

        print(f"PAR: {metrics['PAR']:.2f}% | CAR: {metrics['CAR']:.2f}% | Acc_ALL: {metrics['Acc_ALL']:.2f}%")
        print(f"Memory_Consistency: {metrics['Memory_Consistency']:.2f}% | Memory_Gated_Acc: {metrics['Memory_Gated_Acc']:.2f}%")
        print(f"WeightedScore_WeakPriority: {metrics['WeightedScore_WeakPriority']:.2f}")
        return metrics, type_results, CSV_OUTPUT_PATH

    metrics, raw_results, predictions_csv = evaluate_detailed_v2(model, test_loader, device)

    comparison_row = {
        'run_tag': RUN_TAG,
        'run_profile': RUN_PROFILE,
        'routing': 'weakly_supervised_soft',
        'lambda_prior': float(args.lambda_prior),
        'best_val_acc': float(best_acc),
        'best_epoch': int(best_epoch),
        'Description': float(metrics.get('Description', 0.0)),
        'Explanation': float(metrics.get('Explanation', 0.0)),
        'Predictive-Answer': float(metrics.get('Predictive-Answer', 0.0)),
        'Predictive-Reason': float(metrics.get('Predictive-Reason', 0.0)),
        'Counterfactual-Answer': float(metrics.get('Counterfactual-Answer', 0.0)),
        'Counterfactual-Reason': float(metrics.get('Counterfactual-Reason', 0.0)),
        'PAR': float(metrics.get('PAR', 0.0)),
        'CAR': float(metrics.get('CAR', 0.0)),
        'Acc_ALL': float(metrics.get('Acc_ALL', 0.0)),
        'WeightedScore_WeakPriority': float(metrics.get('WeightedScore_WeakPriority', 0.0)),
    }

    if os.path.exists(COMPARISON_CSV_PATH):
        comp_df = pd.read_csv(COMPARISON_CSV_PATH)
        comp_df = comp_df[comp_df['run_tag'] != RUN_TAG]
        comp_df = pd.concat([comp_df, pd.DataFrame([comparison_row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([comparison_row])

    comp_df = comp_df.sort_values(by='WeightedScore_WeakPriority', ascending=False)
    comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f'Saved/updated run comparison CSV: {COMPARISON_CSV_PATH}')
    print(comp_df[['run_tag', 'run_profile', 'best_val_acc', 'Predictive-Reason', 'Counterfactual-Reason', 'Acc_ALL', 'WeightedScore_WeakPriority']])

    print(metrics)

    # Save final local metrics
    METRICS_JSON_PATH = os.path.join(MODEL_DIR, METRICS_JSON_FILENAME)
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics JSON: {METRICS_JSON_PATH}')
    print('Training/evaluation finished; all outputs are local.')


if __name__ == "__main__":
    main()
