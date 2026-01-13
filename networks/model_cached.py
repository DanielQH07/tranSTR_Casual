"""
Cached version of VideoQAmodel that supports pre-extracted text features.
Import this instead of the original model when using pre-extraction.

Usage:
    from networks.model_cached import VideoQAmodelCached
"""

from builtins import print, tuple
from signal import pause
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

import os
import sys
sys.path.append('../')
from einops import rearrange, repeat
from networks.util import length_to_mask
from networks.multimodal_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from networks.position_encoding import PositionEmbeddingSine1D
from transformers import AutoModel, AutoTokenizer
from networks.topk import HardtopK, PerturbedTopK

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoQAmodelCached(nn.Module):
    """
    VideoQA model with support for pre-extracted text features.
    Use forward_cached() when text features are pre-extracted.
    Use forward() for real-time text encoding (original behavior).
    
    NOTE: This project uses INVERTED mask convention in attention.py:
          True = valid token, False = padding (opposite of PyTorch default)
    """
    
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder=False, n_query=5,
                 objs=20, frames=16, topK_frame=4, topK_obj=5, hard_eval=False, 
                 frame_feat_dim=4096, obj_feat_dim=2053, **kwargs):
        super(VideoQAmodelCached, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.mc = n_query
        self.hard_eval = hard_eval
        
        # Text encoder (only needed for real-time encoding)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # Feature resizers
        self.frame_resize = FeatureResizer(
            input_feat_size=frame_feat_dim,
            output_feat_size=self.d_model,
            dropout=kwargs['dropout'])

        self.obj_resize = FeatureResizer(
            input_feat_size=obj_feat_dim,
            output_feat_size=self.d_model, 
            dropout=kwargs['dropout'])

        self.frame_topK, self.obj_topK = topK_frame, topK_obj
        
        # Text projection (768 -> d_model) for cached features
        self.text_proj = nn.Linear(768, self.d_model)
        
        self.frame_sorter = PerturbedTopK(self.frame_topK)
        self.obj_sorter = PerturbedTopK(self.obj_topK)

        # Transformers
        self.obj_decoder = TransformerDecoder(
            TransformerDecoderLayer(**kwargs), 
            kwargs['num_encoder_layers'],
            norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(
            TransformerDecoderLayer(**kwargs), 
            kwargs['num_encoder_layers'],
            norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(
            TransformerDecoderLayer(**kwargs), 
            kwargs['num_encoder_layers'],
            norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(
            TransformerEncoderLayer(**kwargs), 
            kwargs['num_encoder_layers'],
            norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(
            TransformerDecoderLayer(**kwargs), 
            kwargs['num_encoder_layers'],
            norm=nn.LayerNorm(self.d_model))

        # Position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # Classifier
        self.classifier = nn.Linear(self.d_model, 1)

    def forward(self, frame_feat, obj_feat, qns_word, ans_word):
        """
        Original forward - uses real-time DeBERTa encoding.
        Exactly matches original model.py logic.
        """
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # Encode question
        # Returns: encoded text (projected), attention_mask (True=valid, False=pad)
        q_local, q_mask = self.forward_text(list(qns_word), device)

        # Frame decoder
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(
            frame_feat, q_local,
            memory_key_padding_mask=q_mask,  # True=valid (project convention)
            query_pos=self.pos_encoder_1d(frame_mask, self.d_model),
            output_attentions=True)
        
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)
        else:
            if self.hard_eval:
                idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2)
            else:
                idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)

        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2)

        # Object decoder
        obj_feat = (obj_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B, self.frame_topK, O, -1)
        obj_local = self.obj_resize(obj_feat)
        
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0) if q_mask is not None else None
        
        obj_local, obj_att = self.obj_decoder(
            obj_local.flatten(0,1), q_local_repeated,
            memory_key_padding_mask=q_mask_repeated,
            output_attentions=True)

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
        
        # Overall fusion - exactly like original
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_obj = frame_obj.view(B, self.frame_topK, -1)
        frame_qns_mask = torch.cat((frame_mask, q_mask), dim=1).bool()
        mem = self.vl_encoder(
            torch.cat((frame_obj, q_local), dim=1),
            src_key_padding_mask=frame_qns_mask,
            pos=self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model))
        
        # Answer encoding and decoding
        a_seq, _ = self.forward_text(list(chain(*ans_word)), device, has_ans=True)
        a_seq = rearrange(a_seq, '(n b) t c -> b n t c', b=B)
        tgt = a_seq[:,:,0,:]  # [CLS] token
        out = self.ans_decoder(tgt, mem, memory_key_padding_mask=frame_qns_mask)

        out = self.classifier(out).squeeze(-1)
        return out

    def forward_cached(self, frame_feat, obj_feat, q_encoded, q_mask, qa_encoded, qa_mask):
        """
        Forward with pre-extracted text features.
        
        Args:
            frame_feat: [B, F, frame_feat_dim]
            obj_feat: [B, F, O, obj_feat_dim]
            q_encoded: [B, q_len, 768] - pre-encoded question (raw DeBERTa output)
            q_mask: [B, q_len] - attention mask (True = valid token, False = padding)
            qa_encoded: [B, 5, qa_len, 768] - pre-encoded QA pairs
            qa_mask: [B, 5, qa_len] - QA attention masks (True = valid)
        
        NOTE: Mask convention in this project: True = VALID, False = PADDING
              (attention.py line 91 inverts: ~key_padding_mask)
        """
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        # Resize frame features
        frame_feat = self.frame_resize(frame_feat)
        
        # Project question features (768 -> d_model)
        q_local = self.text_proj(q_encoded)  # [B, q_len, d_model]
        # q_mask is already True=valid, use directly (same as original)

        # Frame decoder
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(
            frame_feat, q_local,
            memory_key_padding_mask=q_mask,  # True=valid
            query_pos=self.pos_encoder_1d(frame_mask, self.d_model),
            output_attentions=True)
        
        if self.training:
            idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)
        else:
            if self.hard_eval:
                idx_frame = rearrange(HardtopK(frame_att.flatten(1,2), self.frame_topK), 'b (f q) k -> b f q k', f=F).sum(-2)
            else:
                idx_frame = rearrange(self.frame_sorter(frame_att.flatten(1,2)), 'b (f q) k -> b f q k', f=F).sum(-2)

        frame_local = (frame_local.transpose(1,2) @ idx_frame).transpose(1,2)

        # Object decoder
        obj_feat = (obj_feat.flatten(-2,-1).transpose(1,2) @ idx_frame).transpose(1,2).view(B, self.frame_topK, O, -1)
        obj_local = self.obj_resize(obj_feat)
        
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0)
        
        obj_local, obj_att = self.obj_decoder(
            obj_local.flatten(0,1), q_local_repeated,
            memory_key_padding_mask=q_mask_repeated,
            output_attentions=True)

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
        
        # Overall fusion - exactly like original
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_obj = frame_obj.view(B, self.frame_topK, -1)
        frame_qns_mask = torch.cat((frame_mask, q_mask), dim=1).bool()
        mem = self.vl_encoder(
            torch.cat((frame_obj, q_local), dim=1),
            src_key_padding_mask=frame_qns_mask,
            pos=self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model))
        
        # Answer decoding with pre-extracted QA features
        # qa_encoded: [B, 5, qa_len, 768], take [CLS] token (position 0)
        qa_cls = qa_encoded[:, :, 0, :]  # [B, 5, 768]
        tgt = self.text_proj(qa_cls)  # [B, 5, d_model]
        out = self.ans_decoder(tgt, mem, memory_key_padding_mask=frame_qns_mask)

        out = self.classifier(out).squeeze(-1)
        return out

    def forward_text(self, text_queries, device, has_ans=False):
        """
        Real-time text encoding using DeBERTa.
        Returns attention_mask with True=valid (project convention).
        """
        tokenized_queries = self.tokenizer.batch_encode_plus(
            text_queries, padding='longest', return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
        
        # Project from 768 to d_model
        encoded_text = self.text_proj(encoded_text)
        
        # Return: encoded text, attention_mask (True = valid token)
        return encoded_text, tokenized_queries.attention_mask.bool()


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


if __name__ == "__main__":
    print("VideoQAmodelCached ready for pre-extracted features")
