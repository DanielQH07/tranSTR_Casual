from builtins import print, tuple
from signal import pause
import torch
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from itertools import chain
# import difftopk

import os
import sys
sys.path.append('../')
from einops import rearrange, repeat
from networks.util import length_to_mask
from networks.multimodal_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from networks.position_encoding import PositionEmbeddingSine1D
from transformers import AutoModel, AutoTokenizer
from networks.topk import HardtopK, PerturbedTopK
from networks.som_injection import SoMInjector

# from networks.encoder import EncoderVid
# from block import fusions #pytorch >= 1.1.0

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class VideoQAmodel(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder = False, n_query=5,
                        objs=20, frames=16, topK_frame=4, topK_obj=5, hard_eval=False, 
                        frame_feat_dim=4096, obj_feat_dim=2053, use_som=False, num_marks=16, **kwargs):
        super(VideoQAmodel, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.mc = n_query
        self.hard_eval = hard_eval
        # text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # Resize frame features to d_model
        self.frame_resize = FeatureResizer(
            input_feat_size=frame_feat_dim,
            output_feat_size=self.d_model,
            dropout=kwargs['dropout'])

        self.obj_resize = FeatureResizer(
            input_feat_size=obj_feat_dim,
            output_feat_size=self.d_model, 
            dropout=kwargs['dropout'])

        self.frame_topK, self.obj_topK = topK_frame, topK_obj
        
        # Add text projection layer (BERT 768 -> d_model)
        self.text_proj = nn.Linear(768, self.d_model)
        self.frame_sorter = PerturbedTopK(self.frame_topK)
        self.obj_sorter = PerturbedTopK(self.obj_topK)

        # hierarchy 1: obj & frame
        self.obj_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # cls head
        self.classifier=nn.Linear(self.d_model, 1) # ans_num+<unk>
        
        # Set-of-Mark Injection (optional)
        self.use_som = use_som
        if use_som:
            self.som_injector = SoMInjector(
                d_model=self.d_model,
                obj_feat_dim=self.d_model,  # All features are in d_model space after resize
                num_marks=num_marks,
                gamma_init=0.1,
                beta_init=0.1
            )
            print(f"[SoM] Enabled with {num_marks} marks (injection after resize)")

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p not in self.text_encoder.parameters():
    #             # if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, frame_feat, obj_feat, qns_word, ans_word, som_data=None):
        """
        :param frame_feat:[bs, T, frame_feat_dim] e.g., [bs, 16, 4096]
        :param obj_feat:[bs, T, O, obj_feat_dim] e.g., [bs, 16, 20, 2053]
        :param qns: ('what are three people sitting on?', 'what is a family having?')
        :param som_data: Optional list of SoM data dicts for Token Mark injection
        :return:
        """
        # Size
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        # Resize frame features to d_model
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # encode q
        q_local, q_mask = self.forward_text(list(qns_word), device)  # [batch, q_len, d_model]


        #### encode v
        # frame
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(frame_feat,
                                    q_local,
                                    memory_key_padding_mask=q_mask,
                                    query_pos = self.pos_encoder_1d(frame_mask , self.d_model),
                                    output_attentions=True
                                    ) # b,16,d
        
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
        else:
            if self.hard_eval:
                idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk
            else:
                idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2) # B*16, O, topk

        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2) # B, Frame_K, d)

        # obj
        obj_feat = (obj_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B,self.frame_topK,O,-1)
        obj_local = self.obj_resize(obj_feat)  # [B, frame_topK, O, d_model]
        
        # Apply SoM injection AFTER resize (so all features are in d_model space)
        if self.use_som and som_data is not None:
            frame_local, obj_local = self.som_injector(
                frame_local, obj_local, som_data, 
                idx_frame=idx_frame  # Pass frame selection indices for proper mapping
            )
        
        # Repeat q_local and q_mask for each frame (handle potential batch size mismatch)
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0) if q_mask is not None else None
        
        obj_local, obj_att = self.obj_decoder(obj_local.flatten(0,1),
                                            q_local_repeated, 
                                            memory_key_padding_mask=q_mask_repeated,
                                            output_attentions=True
                                            )  # b*16,5,d        #.view(B, F, O, -1) # b,16,5,d

        if self.training:
            idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
        else:
            if self.hard_eval:
                idx_obj = rearrange(HardtopK(obj_att.flatten(1,2), self.obj_topK), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
            else:
                idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
        obj_local = (obj_local.transpose(1,2) @ idx_obj).transpose(1,2).view(B, self.frame_topK, self.obj_topK, -1)


        ### hierarchy grouping
        frame_obj = self.fo_decoder(frame_local,
                                    obj_local.flatten(1,2),
                                    # query_pos = self.pos_encoder_1d(frame_mask.view(B,F), self.d_model), \
                                    # memory_key_padding_mask=self.win_mask.unsqueeze(0).repeat(B,1,1).to(device)
                                    ) # b,16,d
        
        ### overall fusion
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_obj =frame_obj.view(B, self.frame_topK, -1)
        frame_qns_mask = torch.cat((frame_mask, q_mask),dim=1).bool()
        mem = self.vl_encoder(torch.cat((frame_obj, q_local), dim=1), \
                            src_key_padding_mask=frame_qns_mask, \
                            pos = self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model)
                            ) # b,16,d
        
        # encode ans
        a_seq, _ = self.forward_text(list(chain(*ans_word)), device, has_ans=True)
        a_seq = rearrange(a_seq, '(n b) t c -> b n t c', b=B)
        tgt = a_seq[:,:,0,:] # [CLS] # [batch, n_query, d_model]
        out = self.ans_decoder(tgt, mem, memory_key_padding_mask=frame_qns_mask)

        # predict
        out = self.classifier(out).squeeze(-1) # 这里squeeze是由于classifier会出来最后一维是1
        return out
    
    def forward_cached(self, frame_feat, obj_feat, text_feat):
        """
        Forward pass using pre-extracted text features (bypasses DeBERTa).
        
        :param frame_feat: [bs, T, frame_feat_dim]
        :param obj_feat: [bs, T, O, obj_feat_dim]
        :param text_feat: [bs, 5, 768] - pre-extracted [CLS] features from DeBERTa
        """
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        # Resize frame features
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # Project cached text features (768 -> d_model)
        # text_feat is [B, 5, 768], we use it as query for questions
        q_local = self.text_proj(text_feat)  # [B, 5, d_model]
        q_mask = torch.zeros(B, q_local.size(1), device=device).bool()  # No mask needed
        
        # Frame decoder with question
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(
            frame_feat,
            q_local,
            memory_key_padding_mask=q_mask,
            query_pos=self.pos_encoder_1d(frame_mask, self.d_model),
            output_attentions=True
        )
        
        # TopK frame selection
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)
        else:
            if self.hard_eval:
                idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2)
            else:
                idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)
        
        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2)
        
        # Object processing
        obj_feat = (obj_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B, self.frame_topK, O, -1)
        obj_local = self.obj_resize(obj_feat)
        
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0)
        
        obj_local, obj_att = self.obj_decoder(
            obj_local.flatten(0,1),
            q_local_repeated,
            memory_key_padding_mask=q_mask_repeated,
            output_attentions=True
        )
        
        # TopK object selection
        if self.training:
            idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2)
        else:
            if self.hard_eval:
                idx_obj = rearrange(HardtopK(obj_att.flatten(1,2), self.obj_topK), 'b (o q) k -> b o q k', o=O).sum(-2)
            else:
                idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2)
        
        obj_local = (obj_local.transpose(1,2) @ idx_obj).transpose(1,2).view(B, self.frame_topK, self.obj_topK, -1)
        
        # Hierarchy grouping
        frame_obj = self.fo_decoder(frame_local, obj_local.flatten(1,2))
        
        # Overall fusion
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_obj = frame_obj.view(B, self.frame_topK, -1)
        frame_qns_mask = torch.cat((frame_mask, q_mask), dim=1).bool()
        
        mem = self.vl_encoder(
            torch.cat((frame_obj, q_local), dim=1),
            src_key_padding_mask=frame_qns_mask,
            pos=self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model)
        )
        
        # Answer decoding - use the cached text features directly as answer targets
        # text_feat is [B, 5, 768], project to d_model
        tgt = q_local  # [B, 5, d_model] - use as answer query
        out = self.ans_decoder(tgt, mem, memory_key_padding_mask=frame_qns_mask)
        
        # Predict
        out = self.classifier(out).squeeze(-1)
        return out
        

    def forward_text(self, text_queries, device, has_ans=False):
        """
        text_queries : list of question str 
        out: text_embedding: bs, len, dim
            mask: bs, len (bool) [1,1,1,1,0,0]
        """
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        # tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='max_length', 
        #                                                     max_length=self.qa_max_len if has_ans else self.q_max_len, 
        #                                                     return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
        
        # Project text from 768 to d_model
        encoded_text = self.text_proj(encoded_text)

        return encoded_text, tokenized_queries.attention_mask.bool()
    


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="train parameter")
    # general
    parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=256)
    parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
    parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=25)
    parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
    parser.add_argument("-es", action="store_true", help="early_stopping")
    parser.add_argument("-dropout", "-drop", type=float, help="dropout rate", default=0.2)  
    parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=5)  
    parser.add_argument("-encoder_dropout", "-ep", type=float, help="dropout rate", default=0.3)   

    # dataset
    parser.add_argument('-dataset', default='msrvtt-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument("-ans_num", type=int, help="ans vocab num", default=5000)
    parser.add_argument("-n_query", type=int, help="multi-choice", default=5)  

    # model
    parser.add_argument("-is_gru", action="store_true", help="gru or lstm in Qns Encoder")
    parser.add_argument("-d_model", "-md",  type=int, help="hidden dim of vq encoder", default=768) 
    parser.add_argument("-word_dim", "-wd", type=int, help="word dim ", default=768)   
    parser.add_argument("-vid_dim", "-vd", type=int, help="vis dim", default=2048) 
    parser.add_argument('-vid_encoder_type', "-ve", default='cnn',choices=['rnn', 'cnn'], type=str)
    parser.add_argument("-hard_eval", "-hd", action="store_true", help="hard selection during inference")
    parser.add_argument("-topK_frame", "-fk", type=int, help="word dim ", default=8)   
    parser.add_argument("-topK_obj", "-ok", type=int, help="word dim ", default=5) 

    # transformer
    parser.add_argument("-trans_hid", type=int, help="hidden dim of ffn in transfomer", default=2048) 
    parser.add_argument("-num_encoder_layers", "-el", type=int, help="number of encoder layers in transformer", default=2)
    parser.add_argument("-num_decoder_layers", "-dl", type=int, help="number of decoder layers in transformer", default=2)
    parser.add_argument("-nheads", type=int, help="num of attention head", default=8) 
    parser.add_argument("-normalize_before", action="store_true", help="pre or post normalize")
    parser.add_argument("-activation", default='relu', choices=['relu','gelu','glu'], type=str)
    parser.add_argument("-return_intermediate", "-ri", action="store_true", help="return intermediate of decoder")
    
    # lan model
    parser.add_argument("-freeze_text_encoder", action="store_true", help="freeze text encoder")
    parser.add_argument("-text_encoder_type", "-t", default="roberta-base", \
                        choices=["roberta-base","distilroberta-base","bert-base-uncased",\
                            "distilbert-base-uncased","microsoft/deberta-base"], type=str)
    parser.add_argument('-text_pool_mode',"-pool", default=0, choices=[0,1,2],help="0last hidden, 1mean, 2max", type=int)

    args = parser.parse_args()
    config = {**vars(args)}
    # print(config)
    # videos=torch.rand(2,8,4096)
    vid_obj_feat = torch.rand(2, 16, 20, 2053)  # [B, T, O, 2048+5]
    vid_frame_feat = torch.rand(2, 16, 4096)    # [B, T, 4096] (app+mot concatenated)
    qns = ('what are three people sitting on?', 'what is a family having?')
    ans = [('how do the two man play the instrument [SEP] roll the handle', \
    'why did the boy pick up one present from the group of them and move to the sofa [SEP] share with the girl'), \
        ('how do the two man play the instrument [SEP] tap their feet', 'why did the boy pick up one present from the group of them and move to the sofa [SEP] approach lady sitting there'), \
    ('how do the two man play the instrument [SEP] strum the string', 'why did the boy pick up one present from the group of them and move to the sofa [SEP] unwrap it'), \
        ('how do the two man play the instrument [SEP] hit with sticks', 'why did the boy pick up one present from the group of them and move to the sofa [SEP] playing with toy train'),\
                ('how do the two man play the instrument [SEP] pat with hand', 'why did the boy pick up one present from the group of them and move to the sofa [SEP] gesture something')]


    model=VideoQAmodel(**config)
    # model.eval()
    model.to('cuda')
    vid_frame_feat = vid_frame_feat.to('cuda')
    vid_obj_feat = vid_obj_feat.to('cuda')
    out = model( vid_frame_feat, vid_obj_feat, qns, ans)
    print(out.shape)

    # parameters
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameter: %.2fM" % (total/1e6))