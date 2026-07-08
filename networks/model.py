from builtins import print, tuple
import math
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
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

# from networks.encoder import EncoderVid
# from block import fusions #pytorch >= 1.1.0

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class ContinuousTimeEncoding(nn.Module):
    """Parameter-free sinusoidal encoding for normalized timestamps in [0, 1]."""

    def __init__(self, d_model, temperature=10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even; got {d_model}")
        self.d_model = int(d_model)
        self.temperature = float(temperature)

    def forward(self, timestamps):
        timestamps = timestamps.clamp(0.0, 1.0) * (2.0 * math.pi)
        dim_t = torch.arange(
            self.d_model, dtype=torch.float32, device=timestamps.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.d_model
        )
        phase = timestamps.float().unsqueeze(-1) / dim_t
        encoding = torch.zeros_like(phase)
        encoding[..., 0::2] = phase[..., 0::2].sin()
        encoding[..., 1::2] = phase[..., 1::2].cos()
        return encoding.to(dtype=timestamps.dtype)


class SparseTemporalRelation(nn.Module):
    """Reason over K selected event tokens and expose selector diagnostics."""

    def __init__(
        self,
        d_model,
        nheads=8,
        dim_feedforward=1024,
        dropout=0.2,
        num_layers=1,
        activation="gelu",
        enabled=True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.time_encoder = ContinuousTimeEncoding(d_model)
        layer = TransformerEncoderLayer(
            d_model=d_model,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    @staticmethod
    def selector_diagnostics(idx_frame):
        # idx_frame: [B, F, K]. Normalize each slot over the original frames.
        selection_prob = idx_frame / idx_frame.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        bsz, num_frames, topk = selection_prob.shape

        time_axis = torch.linspace(
            0.0, 1.0, num_frames,
            device=selection_prob.device,
            dtype=selection_prob.dtype,
        )
        selected_time = torch.einsum(
            "bfk,f->bk", selection_prob, time_axis
        ).clamp(0.0, 1.0)

        slot_prob = F.normalize(
            selection_prob.transpose(1, 2), p=2, dim=-1, eps=1e-6
        )
        gram = torch.bmm(slot_prob, slot_prob.transpose(1, 2))
        if topk > 1:
            eye = torch.eye(
                topk, device=gram.device, dtype=torch.bool
            ).unsqueeze(0)
            overlap = gram.masked_select(~eye).view(
                bsz, topk * (topk - 1)
            ).mean(dim=1)
        else:
            overlap = gram.new_zeros(bsz)

        hard_indices = selection_prob.argmax(dim=1)
        unique_ratio = selection_prob.new_tensor([
            torch.unique(row).numel() / float(topk)
            for row in hard_indices
        ])
        time_span = selected_time.max(dim=1).values - selected_time.min(dim=1).values

        return {
            "frame_selection_prob": selection_prob,
            "selected_time": selected_time,
            "selection_unique_ratio": unique_ratio,
            "selection_overlap": overlap,
            "selected_time_span": time_span,
        }

    def forward(self, event_tokens, idx_frame):
        diagnostics = self.selector_diagnostics(idx_frame)
        if not self.enabled:
            return event_tokens, diagnostics

        time_pos = self.time_encoder(diagnostics["selected_time"]).to(
            dtype=event_tokens.dtype
        )
        event_mask = torch.ones(
            event_tokens.shape[:2],
            dtype=torch.bool,
            device=event_tokens.device,
        )
        event_tokens = self.encoder(
            event_tokens,
            src_key_padding_mask=event_mask,
            pos=time_pos,
        )
        return event_tokens, diagnostics


class SparseFuturePredictor(nn.Module):
    """Predict a future DINO latent from four sparse chronological latents.

    This module is deliberately text/answer agnostic. It is pretrained on
    unique videos, then frozen before being attached to VideoQAmodel.
    """

    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=2,
        nheads=8,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim % nheads != 0:
            raise ValueError(
                f"world hidden_dim ({hidden_dim}) must be divisible by nheads ({nheads})"
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.time_encoder = ContinuousTimeEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.future_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.future_query, std=0.02)
        self.future_attention = nn.MultiheadAttention(
            self.hidden_dim,
            nheads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(
        self,
        context_feat,
        context_time,
        target_time,
        context_valid_mask=None,
    ):
        if context_feat.dim() != 3:
            raise ValueError(
                f"context_feat must be [B,C,D], got {tuple(context_feat.shape)}"
            )
        if context_feat.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected future input dim {self.input_dim}, got {context_feat.size(-1)}"
            )
        if context_time.shape != context_feat.shape[:2]:
            raise ValueError(
                "context_time must match context_feat[:2], got "
                f"{tuple(context_time.shape)} vs {tuple(context_feat.shape[:2])}"
            )
        target_time = target_time.view(-1)
        if target_time.size(0) != context_feat.size(0):
            raise ValueError("target_time must contain one timestamp per sample")

        context = self.input_proj(self.input_norm(context_feat))
        context = context + self.time_encoder(context_time).to(context.dtype)
        padding_mask = None
        if context_valid_mask is not None:
            context_valid_mask = context_valid_mask.bool()
            if context_valid_mask.shape != context_feat.shape[:2]:
                raise ValueError("context_valid_mask must be [B,C]")
            padding_mask = ~context_valid_mask
        context = self.context_encoder(
            context,
            src_key_padding_mask=padding_mask,
        )
        query = self.future_query.expand(context_feat.size(0), -1, -1)
        query = query + self.time_encoder(target_time[:, None]).to(query.dtype)
        future, _ = self.future_attention(
            query,
            context,
            context,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_proj(self.output_norm(future[:, 0]))


class VideoQAmodel(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder = False, n_query=5,
                        objs=20, frames=16, topK_frame=4, topK_obj=5, hard_eval=False, 
                        frame_feat_dim=4096, obj_feat_dim=2053, use_grounding_dino=False,
                        use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                        lora_target_modules=None,
                        obj_use_bbox_pos_embed=True, obj_hard_gather_from_frame=True,
                        obj_bbox_dim=4, obj_split_siglip2=False,
                        obj_roi_dim=768, obj_class_dim=768,
                        use_causal_selector=False, causal_selector_hidden=256,
                        fo_fusion_mode="local", use_sparse_world_model=False,
                        world_context_frames=4, world_hidden_dim=512,
                        world_layers=2, world_nheads=8, world_ffn=1024,
                        world_dropout=0.1,
                        **kwargs):
        super(VideoQAmodel, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.mc = n_query
        self.hard_eval = hard_eval
        self.use_grounding_dino = use_grounding_dino
        self.objs = objs
        self.obj_feat_dim = obj_feat_dim
        self.use_causal_selector = bool(use_causal_selector)
        self.fo_fusion_mode = str(fo_fusion_mode).lower()
        if self.fo_fusion_mode not in {"local", "global"}:
            raise ValueError("fo_fusion_mode must be 'local' or 'global'")
        self.use_sparse_world_model = bool(use_sparse_world_model)
        self.world_context_frames = int(world_context_frames)
        self.world_mode = "full"
        self.world_predictor_frozen = False
        # text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.use_lora = use_lora
        if self.use_lora:
            if get_peft_model is None:
                raise ImportError(
                    "PEFT is required for LoRA. Please install it with: pip install peft"
                )
            if lora_target_modules is None:
                lora_target_modules = ["query_proj", "key_proj", "value_proj"]
            elif isinstance(lora_target_modules, str):
                lora_target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_cfg)

        # Resize frame features to d_model
        self.frame_resize = FeatureResizer(
            input_feat_size=frame_feat_dim,
            output_feat_size=self.d_model,
            dropout=kwargs['dropout'])

        # ---- Object feature path ----
        # With GroundingDINO+SigLIP2 raw features (1540-d = ROI 768 +
        # DeBERTa cls 768 + bbox 4), keep semantic and spatial inputs separate.
        # A single LayerNorm over semantic+bbox features can erase spatial
        # information, so we split:
        #   sem  = [..., :1536]  -> LayerNorm + Linear -> d_model
        #   bbox = [..., -4:]    -> 2D sinusoidal pos embed -> d_model
        # Then sum + LayerNorm. Controlled by obj_use_bbox_pos_embed.
        self.obj_bbox_dim = int(obj_bbox_dim)
        self.obj_use_bbox_pos_embed = bool(obj_use_bbox_pos_embed) and bool(use_grounding_dino)
        self.obj_hard_gather_from_frame = bool(obj_hard_gather_from_frame) and bool(use_grounding_dino)
        self.obj_split_siglip2 = bool(obj_split_siglip2) and self.obj_use_bbox_pos_embed
        self.obj_roi_dim = int(obj_roi_dim)
        self.obj_class_dim = int(obj_class_dim)
        self.obj_expected_dim = int(obj_feat_dim)

        if self.obj_use_bbox_pos_embed:
            sem_dim = int(obj_feat_dim) - self.obj_bbox_dim
            if sem_dim <= 0:
                raise ValueError(f"obj_feat_dim ({obj_feat_dim}) must be > obj_bbox_dim ({self.obj_bbox_dim})")
            if self.d_model % 8 != 0:
                raise ValueError(f"d_model must be divisible by 8 for BBoxPosEmbed2D; got {self.d_model}")
            if self.obj_split_siglip2 and sem_dim != self.obj_roi_dim + self.obj_class_dim:
                raise ValueError(
                    f"SigLIP2 split expects semantic dim {self.obj_roi_dim + self.obj_class_dim}, "
                    f"got {sem_dim} from obj_feat_dim={obj_feat_dim}"
                )
            self.obj_pre_norm = nn.Identity()
            if self.obj_split_siglip2:
                self.obj_roi_norm = nn.LayerNorm(self.obj_roi_dim)
                self.obj_class_norm = nn.LayerNorm(self.obj_class_dim)
                self.obj_roi_resize = FeatureResizer(self.obj_roi_dim, self.d_model, kwargs['dropout'])
                self.obj_class_resize = FeatureResizer(self.obj_class_dim, self.d_model, kwargs['dropout'])
            else:
                self.obj_sem_norm = nn.LayerNorm(sem_dim)
                self.obj_resize = FeatureResizer(sem_dim, self.d_model, kwargs['dropout'])
            self.bbox_pos_embed = BBoxPosEmbed2D(d_model=self.d_model, dropout=kwargs['dropout'])
            self.obj_post_pos_norm = nn.LayerNorm(self.d_model)
        else:
            self.obj_pre_norm = nn.LayerNorm(obj_feat_dim) if use_grounding_dino else nn.Identity()
            self.obj_resize = FeatureResizer(
                input_feat_size=obj_feat_dim,
                output_feat_size=self.d_model, 
                dropout=kwargs['dropout'])

        self.frame_topK, self.obj_topK = topK_frame, topK_obj

        # Sparse temporal reasoning is applied only to the selected K event tokens.
        self.temporal_relation = SparseTemporalRelation(
            d_model=self.d_model,
            nheads=kwargs.get("temporal_relation_nheads", kwargs.get("nheads", 8)),
            dim_feedforward=kwargs.get("temporal_relation_ffn", 1024),
            dropout=kwargs.get("temporal_relation_dropout", 0.2),
            num_layers=kwargs.get("temporal_relation_layers", 1),
            activation=kwargs.get("activation", "gelu"),
            enabled=kwargs.get("use_temporal_relation", True),
        )
        
        # Add text projection layer (BERT 768 -> d_model)
        self.text_proj = nn.Linear(768, self.d_model)
        self.frame_sorter = PerturbedTopK(self.frame_topK)

        # Keep these modules strictly opt-in so a v2 checkpoint can still be
        # strict-loaded when use_causal_selector=False.
        if self.use_causal_selector:
            selector_hidden = int(causal_selector_hidden)
            self.frame_relevance_frame = nn.Linear(self.d_model, selector_hidden)
            self.frame_relevance_question = nn.Linear(self.d_model, selector_hidden)
            self.frame_relevance_head = nn.Linear(selector_hidden, 1)
            # Shared projection prevents the positive/complement comparison
            # from being solved by two unrelated embedding spaces.
            self.causal_frame_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
        
        # obj_sorter only if NOT using GroundingDINO (already filtered by text prompt)
        if not use_grounding_dino:
            self.obj_sorter = PerturbedTopK(self.obj_topK)

        # hierarchy 1: obj & frame
        self.obj_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))

        if self.use_sparse_world_model:
            if self.frame_topK < self.world_context_frames + 1:
                raise ValueError(
                    "Sparse world model needs at least context_frames + 1 selected frames"
                )
            self.world_predictor = SparseFuturePredictor(
                input_dim=frame_feat_dim,
                hidden_dim=world_hidden_dim,
                num_layers=world_layers,
                nheads=world_nheads,
                dim_feedforward=world_ffn,
                dropout=world_dropout,
            )
            self.world_feature_resize = FeatureResizer(
                input_feat_size=frame_feat_dim,
                output_feat_size=self.d_model,
                dropout=0.0,
            )
            self.world_feature_resize.load_state_dict(
                self.frame_resize.state_dict()
            )
            self.world_question_proj = nn.Linear(self.d_model, self.d_model)
            self.world_object_proj = nn.Linear(self.d_model, self.d_model)
            self.world_event_proj = nn.Linear(self.d_model, self.d_model)
            self.world_delta = nn.Sequential(
                nn.Linear(self.d_model * 3, world_hidden_dim),
                nn.GELU(),
                nn.Linear(world_hidden_dim, frame_feat_dim),
            )
            nn.init.zeros_(self.world_delta[-1].weight)
            nn.init.zeros_(self.world_delta[-1].bias)
            self.world_cause_query = nn.Linear(
                self.d_model, self.d_model, bias=False
            )
            self.world_cause_key = nn.Linear(
                self.d_model, self.d_model, bias=False
            )
            self.world_type_embedding = nn.Parameter(
                torch.zeros(3, self.d_model)
            )
            nn.init.normal_(self.world_type_embedding, std=0.02)
            self.world_gate_head = nn.Linear(self.d_model, 2)
            nn.init.zeros_(self.world_gate_head.weight)
            nn.init.constant_(self.world_gate_head.bias, -2.0)
            self.world_decoder = TransformerDecoder(
                TransformerDecoderLayer(**kwargs),
                1,
                norm=nn.LayerNorm(self.d_model),
            )
            self.world_output_proj = nn.Linear(self.d_model, self.d_model)
            nn.init.zeros_(self.world_output_proj.weight)
            nn.init.zeros_(self.world_output_proj.bias)

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # Generalized scoring heads
        self.answer_head = nn.Linear(self.d_model, 1)
        self.evidence_head = nn.Linear(self.d_model, 1)

        # Family-aware unified knowledge verifier
        num_question_families = kwargs.get("num_question_families", 6)
        self.q_family_embed = nn.Embedding(num_embeddings=num_question_families, embedding_dim=self.d_model)
        self.knowledge_head = nn.Sequential(
            nn.Linear(self.d_model * 6, self.d_model),
            nn.ReLU(),
            nn.Dropout(kwargs['dropout']),
            nn.Linear(self.d_model, 1)
        )
        self.k_proj = nn.Linear(kwargs.get("knowledge_feat_dim", self.d_model), self.d_model)

        # Training weights for evidence/knowledge losses
        self.lambda_evidence = kwargs.get("lambda_evidence", 0.3)
        self.lambda_knowledge = kwargs.get("lambda_knowledge", 0.2)

        # Backward-compat aliases for old training scripts
        self.classifier = self.answer_head
        self.verifier = self.evidence_head

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p not in self.text_encoder.parameters():
    #             # if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def decode_candidates(self, cand_feat):
        answer_score = self.answer_head(cand_feat).squeeze(-1)
        evidence_score = self.evidence_head(cand_feat).squeeze(-1)
        return cand_feat, answer_score, evidence_score

    def pool_memory(self, mem, mem_mask=None):
        # Mean-pool by default to keep behavior stable and lightweight.
        if mem_mask is None:
            return mem.mean(dim=1)

        # mem_mask shape [B, L]. If all tokens are marked valid, this reduces to normal mean.
        valid = mem_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1e-6)
        return (mem * valid).sum(dim=1) / denom

    def score_knowledge_support(self, cand_feat, mem_pool, k_feat, q_family_id):
        qf = self.q_family_embed(q_family_id)
        qf = qf.unsqueeze(1).expand(-1, cand_feat.size(1), -1)

        mem_expand = mem_pool.unsqueeze(1).expand(-1, cand_feat.size(1), -1)

        feat = torch.cat([
            cand_feat,
            mem_expand,
            k_feat,
            cand_feat * mem_expand,
            cand_feat * k_feat,
            qf,
        ], dim=-1)

        knowledge_score = self.knowledge_head(feat).squeeze(-1)
        return knowledge_score

    def _normalize_knowledge_feat(self, knowledge_feat, cand_feat):
        """Normalize arbitrary knowledge features to [B, N, d_model]."""
        bsz, num_cands = cand_feat.shape[:2]
        device = cand_feat.device

        if knowledge_feat is None:
            return torch.zeros(bsz, num_cands, self.d_model, device=device)

        if not isinstance(knowledge_feat, torch.Tensor):
            knowledge_feat = torch.tensor(knowledge_feat, dtype=cand_feat.dtype, device=device)

        knowledge_feat = knowledge_feat.to(device)
        if knowledge_feat.dim() == 2:
            # [B, Dk] -> [B, N, Dk]
            knowledge_feat = knowledge_feat.unsqueeze(1).expand(-1, num_cands, -1)
        elif knowledge_feat.dim() == 3:
            # [B, Nk, Dk] -> [B, N, Dk]
            if knowledge_feat.size(1) == 1:
                knowledge_feat = knowledge_feat.expand(-1, num_cands, -1)
            elif knowledge_feat.size(1) != num_cands:
                if knowledge_feat.size(1) > num_cands:
                    knowledge_feat = knowledge_feat[:, :num_cands, :]
                else:
                    repeat_times = (num_cands + knowledge_feat.size(1) - 1) // knowledge_feat.size(1)
                    knowledge_feat = knowledge_feat.repeat(1, repeat_times, 1)[:, :num_cands, :]
        else:
            raise ValueError("knowledge_feat must be rank-2 or rank-3 tensor")

        if knowledge_feat.size(-1) != self.d_model:
            knowledge_feat = self.k_proj(knowledge_feat)

        return knowledge_feat

    def _fit_obj_feat_dim(self, obj_feat):
        """Pad or truncate object features to the width expected downstream.
        When obj_use_bbox_pos_embed is True, target = sem_dim + bbox_dim (e.g. 1536+4=1540);
        otherwise target = obj_resize.fc.in_features.
        """
        expected_dim = self.obj_expected_dim
        current_dim = obj_feat.size(-1)
        if current_dim == expected_dim:
            return obj_feat
        if current_dim > expected_dim:
            return obj_feat[..., :expected_dim]
        return F.pad(obj_feat, (0, expected_dim - current_dim))

    def _encode_objects(self, obj_feat):
        """Convert raw object features to obj_local of dim d_model.
        Splits bbox into a 2D sinusoidal positional embedding when
        obj_use_bbox_pos_embed is True; otherwise uses a single LN+Linear path.
        """
        if self.obj_use_bbox_pos_embed:
            sem = obj_feat[..., :-self.obj_bbox_dim]
            bbox = obj_feat[..., -self.obj_bbox_dim:]
            padding_mask = sem.abs().sum(dim=-1).eq(0)
            if self.obj_split_siglip2:
                roi = sem[..., :self.obj_roi_dim]
                class_text = sem[..., self.obj_roi_dim:self.obj_roi_dim + self.obj_class_dim]
                roi_proj = self.obj_roi_resize(self.obj_roi_norm(roi))
                class_proj = self.obj_class_resize(self.obj_class_norm(class_text))
                sem_proj = (roi_proj + class_proj) / math.sqrt(2.0)
            else:
                sem_proj = self.obj_resize(self.obj_sem_norm(sem))
            encoded = self.obj_post_pos_norm(sem_proj + self.bbox_pos_embed(bbox))
            return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask
        padding_mask = obj_feat.abs().sum(dim=-1).eq(0)
        encoded = self.obj_resize(self.obj_pre_norm(obj_feat))
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask

    @staticmethod
    def _valid_attention_mask(padding_mask):
        # Custom MultiheadAttention uses True=valid (opposite to PyTorch padding masks).
        valid_mask = ~padding_mask
        empty_rows = ~valid_mask.any(dim=-1)
        if empty_rows.any():
            valid_mask[empty_rows, 0] = True
        return valid_mask

    def freeze_world_predictor(self):
        if not self.use_sparse_world_model:
            raise RuntimeError("Sparse world model is disabled")
        for parameter in self.world_predictor.parameters():
            parameter.requires_grad_(False)
        self.world_predictor_frozen = True
        self.world_predictor.eval()

    def load_world_predictor(self, state_dict, strict=True, freeze=True):
        if not self.use_sparse_world_model:
            raise RuntimeError("Sparse world model is disabled")
        result = self.world_predictor.load_state_dict(state_dict, strict=strict)
        if freeze:
            self.freeze_world_predictor()
        return result

    def set_world_mode(self, mode):
        mode = str(mode).lower()
        if mode not in {"off", "factual", "full"}:
            raise ValueError("world mode must be 'off', 'factual', or 'full'")
        self.world_mode = mode

    def train(self, mode=True):
        super().train(mode)
        if (
            getattr(self, "use_sparse_world_model", False)
            and getattr(self, "world_predictor_frozen", False)
        ):
            self.world_predictor.eval()
        return self

    @staticmethod
    def _masked_mean(tokens, valid_mask):
        """Mean-pool tokens where True means valid."""
        if valid_mask is None:
            return tokens.mean(dim=1)
        valid = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _normalize_selection(selection):
        return selection / selection.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _hard_positive_complement_overlap(positive, complement):
        """Fraction of positive hard frame indices also in the complement."""
        positive_idx = positive.argmax(dim=1)
        complement_idx = complement.argmax(dim=1)
        overlap = (
            positive_idx.unsqueeze(-1) == complement_idx.unsqueeze(1)
        ).any(dim=-1)
        return overlap.float().mean(dim=1)

    def _hard_causal_topk(self, relevance_score):
        """Deterministic disjoint Top-K and Bottom-K, including score ties."""
        positive = HardtopK(relevance_score, self.frame_topK)
        positive_mask = positive.sum(dim=-1).bool()
        complement_score = (-relevance_score).masked_fill(
            positive_mask, torch.finfo(relevance_score.dtype).min
        )
        complement = HardtopK(complement_score, self.frame_topK)
        return positive, complement

    def _select_frame_evidence(
        self, frame_local, frame_att, q_local, q_mask,
        causal_frame_source=None,
    ):
        """Select positive Top-K frames and optional causal complement.

        The legacy path is intentionally byte-for-byte equivalent in its
        selector computation. The causal path scores each frame with a scalar
        conditioned on a masked global question representation.
        """
        num_frames = frame_local.size(1)
        causal_aux = {}

        if not self.use_causal_selector:
            legacy_score = frame_att.flatten(1, 2)
            if self.training or not self.hard_eval:
                flat_selection = self.frame_sorter(legacy_score)
            else:
                flat_selection = HardtopK(legacy_score, self.frame_topK)
            positive = rearrange(
                flat_selection,
                'b (f q) k -> b f q k',
                f=num_frames,
            ).sum(-2)
            selected_frames = (
                frame_local.transpose(1, 2) @ positive
            ).transpose(1, 2)
            return selected_frames, positive, causal_aux

        # The causal branch must not update the text encoder. It learns the
        # question projection from detached text features and the visual path
        # from the resized frame source. The answer path below still uses the
        # question-conditioned frame_decoder output normally.
        q_global = self._masked_mean(q_local.detach(), q_mask)
        if causal_frame_source is None:
            causal_frame_source = frame_local
        relevance_score = self.frame_relevance_head(
            torch.tanh(
                self.frame_relevance_frame(causal_frame_source)
                + self.frame_relevance_question(q_global).unsqueeze(1)
            )
        ).squeeze(-1)

        if self.training or not self.hard_eval:
            positive = self.frame_sorter(relevance_score)
            complement = self.frame_sorter(-relevance_score)
        else:
            positive, complement = self._hard_causal_topk(relevance_score)

        selected_frames = (
            frame_local.transpose(1, 2) @ positive
        ).transpose(1, 2)
        complement_frames = (
            causal_frame_source.transpose(1, 2) @ complement
        ).transpose(1, 2)
        positive_causal_frames = (
            causal_frame_source.transpose(1, 2) @ positive
        ).transpose(1, 2)

        positive_embedding = F.normalize(
            self.causal_frame_projection(positive_causal_frames.mean(dim=1)),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        complement_embedding = F.normalize(
            self.causal_frame_projection(complement_frames.mean(dim=1)),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        causal_aux = {
            "frame_relevance_score": relevance_score,
            "complement_selection_prob": self._normalize_selection(complement),
            "causal_positive_embedding": positive_embedding,
            "causal_complement_embedding": complement_embedding,
            "positive_complement_overlap": self._hard_positive_complement_overlap(
                positive, complement
            ),
        }
        return selected_frames, positive, causal_aux

    def _select_obj_by_frame(self, obj_feat, idx_frame):
        """Pick objects of the top-K selected frames.
        - Hard path (obj_hard_gather_from_frame=True): gather using argmax of
          idx_frame; preserves per-frame spatial correspondence (essential for
          GroundingDINO bbox features).
        - Soft path: original soft-mix via matmul (mixes objects across frames).
        idx_frame: [B, F, K], obj_feat: [B, F, O, D] -> returns [B, K, O, D].
        """
        B, F, O, D = obj_feat.shape
        K = self.frame_topK
        if self.obj_hard_gather_from_frame:
            with torch.no_grad():
                sel = idx_frame.argmax(dim=1)  # [B, K]
            sel_exp = sel[:, :, None, None].expand(B, K, O, D)
            return torch.gather(obj_feat, dim=1, index=sel_exp)
        # Soft-mix (legacy): obj_feat[b,f,o,d] @ idx_frame[b,f,k] -> [b,k,o,d]
        return (obj_feat.flatten(-2, -1).transpose(1, 2) @ idx_frame).transpose(1, 2).view(B, K, O, D)

    def _fuse_frame_objects_local(self, frame_local, obj_local, obj_padding_mask=None):
        """Fuse each selected frame only with objects from that same frame.

        K is folded into the batch, so fo_decoder cannot mix objects across
        events. Cross-event interaction happens later in temporal_relation.
        """
        if frame_local.dim() != 3 or obj_local.dim() != 4:
            raise ValueError(
                "Local fusion expects frame_local [B,K,D] and obj_local [B,K,O,D], "
                f"got {tuple(frame_local.shape)} and {tuple(obj_local.shape)}"
            )

        B, K, D = frame_local.shape
        obj_B, obj_K, O, obj_D = obj_local.shape
        if (obj_B, obj_K, obj_D) != (B, K, D):
            raise ValueError(
                "Frame/object dimensions are not aligned for local fusion: "
                f"frame={tuple(frame_local.shape)}, object={tuple(obj_local.shape)}"
            )

        frame_query = frame_local.reshape(B * K, 1, D)
        object_memory = obj_local.reshape(B * K, O, D)
        memory_valid_mask = None
        if obj_padding_mask is not None:
            if tuple(obj_padding_mask.shape) != (B, K, O):
                raise ValueError(
                    f"obj_padding_mask must be [B,K,O], got {tuple(obj_padding_mask.shape)}"
                )
            memory_valid_mask = self._valid_attention_mask(
                obj_padding_mask.reshape(B * K, O)
            )

        event_tokens = self.fo_decoder(
            frame_query,
            object_memory,
            memory_key_padding_mask=memory_valid_mask,
        )
        return event_tokens.reshape(B, K, D)

    def _fuse_frame_objects(self, frame_local, obj_local, obj_padding_mask=None):
        if self.fo_fusion_mode == "local":
            return self._fuse_frame_objects_local(
                frame_local,
                obj_local,
                obj_padding_mask=obj_padding_mask,
            )

        # v1 behavior: every selected frame query can attend to every selected
        # object. The custom attention mask uses True=valid.
        memory_valid_mask = None
        if obj_padding_mask is not None:
            memory_valid_mask = self._valid_attention_mask(
                obj_padding_mask.flatten(1, 2)
            )
        return self.fo_decoder(
            frame_local,
            obj_local.flatten(1, 2),
            memory_key_padding_mask=memory_valid_mask,
        )

    def _apply_sparse_world_model(
        self,
        candidate_tgt,
        baseline_out,
        q_local,
        q_mask,
        frame_feat_raw,
        obj_local,
        obj_padding_mask,
        event_tokens,
        idx_frame,
        world_mode=None,
        world_question_permutation=None,
    ):
        if not self.use_sparse_world_model:
            return baseline_out, {}
        if self.training and not self.world_predictor_frozen:
            raise RuntimeError(
                "Freeze the pretrained world_predictor before QA training"
            )

        mode = self.world_mode if world_mode is None else str(world_mode).lower()
        if mode not in {"off", "factual", "full"}:
            raise ValueError("world_mode must be 'off', 'factual', or 'full'")
        batch_size, num_frames = frame_feat_raw.shape[:2]
        topk = idx_frame.size(-1)
        context_len = self.world_context_frames
        if topk < context_len + 1:
            raise ValueError("Not enough selected frames for sparse world rollout")

        q_global = self._masked_mean(q_local, q_mask)
        world_q = q_global
        if world_question_permutation is not None:
            permutation = world_question_permutation.to(q_global.device).long()
            if permutation.shape != (batch_size,):
                raise ValueError("world_question_permutation must be [B]")
            world_q = q_global[permutation]

        # World rollout is intentionally detached from the frame selector.
        hard_indices = idx_frame.detach().argmax(dim=1)  # [B,K]
        gather_raw = hard_indices.unsqueeze(-1).expand(
            -1, -1, frame_feat_raw.size(-1)
        )
        selected_raw = torch.gather(frame_feat_raw, 1, gather_raw)
        selected_time = hard_indices.to(frame_feat_raw.dtype) / max(
            float(num_frames - 1), 1.0
        )
        order = selected_time.argsort(dim=1)

        def gather_slots(tensor):
            index = order
            for _ in range(tensor.dim() - 2):
                index = index.unsqueeze(-1)
            index = index.expand(-1, -1, *tensor.shape[2:])
            return torch.gather(tensor, 1, index)

        selected_raw = gather_slots(selected_raw)
        selected_time = torch.gather(selected_time, 1, order)
        sorted_objects = gather_slots(obj_local)
        sorted_obj_padding = gather_slots(obj_padding_mask)
        sorted_events = gather_slots(event_tokens)

        context_raw = selected_raw[:, :context_len]
        context_time = selected_time[:, :context_len]
        target_time = selected_time[:, context_len]
        context_objects = sorted_objects[:, :context_len]
        context_obj_padding = sorted_obj_padding[:, :context_len]
        context_events = sorted_events[:, :context_len]

        if self.world_predictor_frozen:
            self.world_predictor.eval()
        with torch.no_grad():
            factual_raw = self.world_predictor(
                context_raw,
                context_time,
                target_time,
            )

        q_query = self.world_question_proj(world_q)
        object_key = self.world_object_proj(context_objects)
        object_logits = torch.einsum(
            "bd,bcod->bco", q_query, object_key
        ) / math.sqrt(float(self.d_model))
        object_valid = ~context_obj_padding
        object_mask = torch.sigmoid(object_logits) * object_valid.to(
            object_logits.dtype
        )
        object_denom = object_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        object_summary = (
            context_objects * object_mask.unsqueeze(-1)
        ).sum(dim=2) / object_denom

        event_key = self.world_event_proj(context_events)
        time_logits = torch.einsum(
            "bd,bcd->bc", q_query, event_key
        ) / math.sqrt(float(self.d_model))
        time_mask = torch.sigmoid(time_logits)
        q_expand = world_q.unsqueeze(1).expand(-1, context_len, -1)
        delta_raw = self.world_delta(
            torch.cat((context_events, object_summary, q_expand), dim=-1)
        )
        intervened_context = context_raw + time_mask.unsqueeze(-1) * delta_raw
        counterfactual_raw = self.world_predictor(
            intervened_context,
            context_time,
            target_time,
        )

        future_pair = torch.stack((factual_raw, counterfactual_raw), dim=1)
        # A deterministic projection is essential: independent dropout masks
        # would create a fake factual/counterfactual difference at zero delta.
        future_pair = self.world_feature_resize(future_pair)
        factual_token = future_pair[:, 0]
        counterfactual_token = future_pair[:, 1]
        change_token = counterfactual_token - factual_token
        cause_logits = torch.einsum(
            "bd,bkd->bk",
            self.world_cause_query(change_token),
            self.world_cause_key(sorted_events),
        ) / math.sqrt(float(self.d_model))
        cause_attention = torch.softmax(cause_logits, dim=1)
        cause_token = torch.einsum(
            "bk,bkd->bd", cause_attention, sorted_events
        )

        gates = torch.sigmoid(self.world_gate_head(world_q))
        aux = {
            "world_factual_embedding": factual_token,
            "world_counterfactual_embedding": counterfactual_token,
            "world_object_intervention_mask": object_mask,
            "world_time_intervention_mask": time_mask,
            "world_gate": gates,
            "world_delta_norm": delta_raw.norm(dim=-1).mean(dim=1),
            "world_cause_attention": cause_attention,
            "world_selected_frame_indices": hard_indices,
        }
        if mode == "off":
            return baseline_out, aux

        if mode == "factual":
            world_memory = (
                gates[:, 0, None] * factual_token
                + self.world_type_embedding[0]
            ).unsqueeze(1)
        else:
            world_memory = torch.stack(
                (
                    gates[:, 0, None] * factual_token
                    + self.world_type_embedding[0],
                    gates[:, 1, None] * counterfactual_token
                    + self.world_type_embedding[1],
                    gates[:, 1, None] * cause_token
                    + self.world_type_embedding[2],
                ),
                dim=1,
            )
        world_candidate = self.world_decoder(candidate_tgt, world_memory)
        world_residual = self.world_output_proj(world_candidate)
        aux["world_candidate_residual_norm"] = world_residual.norm(
            dim=-1
        ).mean(dim=1)
        return baseline_out + world_residual, aux

    def forward_with_knowledge(self, frame_feat, obj_feat, qns_word, ans_word, q_family_id, knowledge_feat=None):
        """Knowledge-aware forward that returns detailed scores for reranking/training."""
        aux = self.forward(
            frame_feat,
            obj_feat,
            qns_word,
            ans_word,
            return_aux=True,
            q_family_id=q_family_id,
            knowledge_feat=knowledge_feat,
        )
        if "knowledge_score" in aux:
            aux["fused_score"] = aux["answer_score"] + self.lambda_knowledge * aux["knowledge_score"]
        else:
            aux["fused_score"] = aux["answer_score"]
        return aux

    def forward(
        self, frame_feat, obj_feat, qns_word, ans_word, return_aux=False,
        q_family_id=None, knowledge_feat=None, world_mode=None,
        world_question_permutation=None,
    ):
        """
        :param frame_feat:[bs, T, frame_feat_dim] e.g., [bs, 16, 4096]
        :param obj_feat:[bs, T, O, obj_feat_dim] e.g., [bs, 16, 20, 2053]
        :param qns: ('what are three people sitting on?', 'what is a family having?')
        :return:
        """
        # Size
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        frame_feat_raw = frame_feat
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
        
        frame_local, idx_frame, causal_aux = self._select_frame_evidence(
            frame_local, frame_att, q_local, q_mask,
            causal_frame_source=frame_feat,
        )

        # obj: hard-gather (preserves spatial correspondence) or legacy soft-mix
        obj_feat = self._select_obj_by_frame(obj_feat, idx_frame)
        obj_feat = self._fit_obj_feat_dim(obj_feat)
        obj_local, obj_padding_mask = self._encode_objects(obj_feat)
        obj_decoder_valid_mask = self._valid_attention_mask(obj_padding_mask.flatten(0, 1))
        
        # Repeat q_local and q_mask for each frame (handle potential batch size mismatch)
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0) if q_mask is not None else None
        
        obj_local, obj_att = self.obj_decoder(obj_local.flatten(0,1),
                                            q_local_repeated,
                                            tgt_key_padding_mask=obj_decoder_valid_mask,
                                            memory_key_padding_mask=q_mask_repeated,
                                            output_attentions=True
                                            )  # b*16,O,d        #.view(B, F, O, -1) # b,16,O,d
        obj_local = obj_local.masked_fill(
            obj_padding_mask.flatten(0, 1).unsqueeze(-1), 0.0
        )

        # GroundingDINO: skip obj_sorter (already filtered by text prompt)
        if self.use_grounding_dino:
            # Use all objects (already relevant from text-prompted detection)
            obj_local = obj_local.view(B, self.frame_topK, O, -1)  # [B, frame_topK, objs, d_model]
            fo_obj_padding_mask = obj_padding_mask
        else:
            # Original: topK object selection via attention
            if self.training:
                idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
            else:
                if self.hard_eval:
                    idx_obj = rearrange(HardtopK(obj_att.flatten(1,2), self.obj_topK), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
                else:
                    idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2) # B*frame_topK, O, obj_topk
            obj_local = (obj_local.transpose(1,2) @ idx_obj).transpose(1,2).view(B, self.frame_topK, self.obj_topK, -1)
            fo_obj_padding_mask = None


        ### hierarchy grouping
        frame_obj = self._fuse_frame_objects(
            frame_local,
            obj_local,
            obj_padding_mask=fo_obj_padding_mask,
        )
        
        ### sparse temporal relation reasoning over the selected events
        frame_obj = frame_obj.view(B, self.frame_topK, -1)
        frame_obj, selector_aux = self.temporal_relation(frame_obj, idx_frame)

        ### overall fusion
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_qns_mask = torch.cat((frame_mask, q_mask),dim=1).bool()
        mem = self.vl_encoder(torch.cat((frame_obj, q_local), dim=1), \
                            src_key_padding_mask=frame_qns_mask, \
                            pos = self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model)
                            ) # b,16,d
        
        # encode ans
        a_seq, _ = self.forward_text(list(chain(*ans_word)), device, has_ans=True)
        a_seq = rearrange(a_seq, '(n b) t c -> b n t c', b=B)
        tgt = a_seq[:,:,0,:] # [CLS] # [batch, n_query, d_model]
        if self.use_causal_selector:
            causal_aux["causal_candidate_embedding"] = torch.nn.functional.normalize(
                tgt.detach(), p=2, dim=-1, eps=1e-6
            )
        baseline_out = self.ans_decoder(
            tgt, mem, memory_key_padding_mask=frame_qns_mask
        )
        world_obj_padding = fo_obj_padding_mask
        if world_obj_padding is None:
            world_obj_padding = torch.zeros(
                obj_local.shape[:3], dtype=torch.bool, device=obj_local.device
            )
        out, world_aux = self._apply_sparse_world_model(
            candidate_tgt=tgt,
            baseline_out=baseline_out,
            q_local=q_local,
            q_mask=q_mask,
            frame_feat_raw=frame_feat_raw,
            obj_local=obj_local,
            obj_padding_mask=world_obj_padding,
            event_tokens=frame_obj,
            idx_frame=idx_frame,
            world_mode=world_mode,
            world_question_permutation=world_question_permutation,
        )

        # candidate decoding
        cand_feat, answer_score, evidence_score = self.decode_candidates(out)
        mem_pool = self.pool_memory(mem, mem_mask=frame_qns_mask)

        # Backward-compatible names
        logits = answer_score
        verifier_logits = evidence_score
        aux = {
                "cand_feat": cand_feat,
                "answer_score": answer_score,
                "evidence_score": evidence_score,
                "mem": mem,
                "mem_pool": mem_pool,
                "logits": logits,
                "verifier_logits": verifier_logits,
                **selector_aux,
                **causal_aux,
                **world_aux,
            }

        if q_family_id is not None:
            if not isinstance(q_family_id, torch.Tensor):
                q_family_id = torch.tensor(q_family_id, dtype=torch.long, device=logits.device)
            q_family_id = q_family_id.to(logits.device).long().view(-1)
            k_feat = self._normalize_knowledge_feat(knowledge_feat, cand_feat)
            knowledge_score = self.score_knowledge_support(cand_feat, mem_pool, k_feat, q_family_id)
            aux["knowledge_score"] = knowledge_score
            aux["fused_score"] = answer_score + self.lambda_knowledge * knowledge_score

        if return_aux:
            return aux
        return logits
    
    def forward_cached(
        self, frame_feat, obj_feat, text_feat, return_aux=False,
        q_family_id=None, knowledge_feat=None, question_feat=None,
        question_mask=None, world_mode=None,
        world_question_permutation=None,
    ):
        """
        Forward pass using pre-extracted text features (bypasses DeBERTa).
        
        :param frame_feat: [bs, T, frame_feat_dim]
        :param obj_feat: [bs, T, O, obj_feat_dim]
        :param text_feat: [bs, 5, 768] - pre-extracted [CLS] features for each QA pair
        """
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        frame_feat_raw = frame_feat
        # Resize frame features
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # Project cached text features (768 -> d_model)
        text_feat_proj = self.text_proj(text_feat)  # [B, 5, d_model]
        
        if question_feat is not None:
            question_feat = question_feat.to(device)
            q_local = self.text_proj(question_feat)
            q_mask = (
                question_mask.to(device).bool()
                if question_mask is not None
                else torch.ones(
                    q_local.shape[:2], device=device, dtype=torch.bool
                )
            )
        else:
            if self.use_sparse_world_model:
                raise ValueError(
                    "forward_cached requires question_feat when sparse world model is enabled"
                )
            # Legacy fallback for checkpoints without the world branch.
            q_local = text_feat_proj.mean(dim=1, keepdim=True)
            q_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
        
        # Frame decoder with question
        frame_mask = torch.ones(B, F).bool().to(device)
        frame_local, frame_att = self.frame_decoder(
            frame_feat,
            q_local,
            memory_key_padding_mask=q_mask,
            query_pos=self.pos_encoder_1d(frame_mask, self.d_model),
            output_attentions=True
        )
        
        frame_local, idx_frame, causal_aux = self._select_frame_evidence(
            frame_local, frame_att, q_local, q_mask,
            causal_frame_source=frame_feat,
        )
        
        # Object processing (hard-gather + bbox pos-embed when enabled)
        obj_feat = self._select_obj_by_frame(obj_feat, idx_frame)
        obj_feat = self._fit_obj_feat_dim(obj_feat)
        obj_local, obj_padding_mask = self._encode_objects(obj_feat)
        obj_decoder_valid_mask = self._valid_attention_mask(obj_padding_mask.flatten(0, 1))
        
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0)
        
        obj_local, obj_att = self.obj_decoder(
            obj_local.flatten(0,1),
            q_local_repeated,
            tgt_key_padding_mask=obj_decoder_valid_mask,
            memory_key_padding_mask=q_mask_repeated,
            output_attentions=True
        )
        obj_local = obj_local.masked_fill(
            obj_padding_mask.flatten(0, 1).unsqueeze(-1), 0.0
        )
        
        if self.use_grounding_dino:
            obj_local = obj_local.view(B, self.frame_topK, O, -1)
            fo_obj_padding_mask = obj_padding_mask
        else:
            # TopK object selection
            if self.training:
                idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2)
            else:
                if self.hard_eval:
                    idx_obj = rearrange(HardtopK(obj_att.flatten(1,2), self.obj_topK), 'b (o q) k -> b o q k', o=O).sum(-2)
                else:
                    idx_obj = rearrange(self.obj_sorter(obj_att.flatten(1,2)), 'b (o q) k -> b o q k', o=O).sum(-2)
            
            obj_local = (obj_local.transpose(1,2) @ idx_obj).transpose(1,2).view(B, self.frame_topK, self.obj_topK, -1)
            fo_obj_padding_mask = None
        
        # Hierarchy grouping
        frame_obj = self._fuse_frame_objects(
            frame_local,
            obj_local,
            obj_padding_mask=fo_obj_padding_mask,
        )
        
        # Sparse temporal relation reasoning over the selected events
        frame_obj = frame_obj.view(B, self.frame_topK, -1)
        frame_obj, selector_aux = self.temporal_relation(frame_obj, idx_frame)

        # Overall fusion
        frame_mask = torch.ones(B, self.frame_topK).bool().to(device)
        frame_qns_mask = torch.cat((frame_mask, q_mask), dim=1).bool()
        
        mem = self.vl_encoder(
            torch.cat((frame_obj, q_local), dim=1),
            src_key_padding_mask=frame_qns_mask,
            pos=self.pos_encoder_1d(frame_qns_mask.bool(), self.d_model)
        )
        
        # Answer decoding - use individual QA features as answer queries
        # text_feat_proj is [B, 5, d_model] - each is [CLS] of "question + answer_i"
        tgt = text_feat_proj  # [B, 5, d_model]
        if self.use_causal_selector:
            causal_aux["causal_candidate_embedding"] = torch.nn.functional.normalize(
                tgt.detach(), p=2, dim=-1, eps=1e-6
            )
        baseline_out = self.ans_decoder(
            tgt, mem, memory_key_padding_mask=frame_qns_mask
        )
        world_obj_padding = fo_obj_padding_mask
        if world_obj_padding is None:
            world_obj_padding = torch.zeros(
                obj_local.shape[:3], dtype=torch.bool, device=obj_local.device
            )
        out, world_aux = self._apply_sparse_world_model(
            candidate_tgt=tgt,
            baseline_out=baseline_out,
            q_local=q_local,
            q_mask=q_mask,
            frame_feat_raw=frame_feat_raw,
            obj_local=obj_local,
            obj_padding_mask=world_obj_padding,
            event_tokens=frame_obj,
            idx_frame=idx_frame,
            world_mode=world_mode,
            world_question_permutation=world_question_permutation,
        )
        
        # candidate decoding
        cand_feat, answer_score, evidence_score = self.decode_candidates(out)
        mem_pool = self.pool_memory(mem, mem_mask=frame_qns_mask)

        # Backward-compatible names
        logits = answer_score
        verifier_logits = evidence_score
        aux = {
                "cand_feat": cand_feat,
                "answer_score": answer_score,
                "evidence_score": evidence_score,
                "mem": mem,
                "mem_pool": mem_pool,
                "logits": logits,
                "verifier_logits": verifier_logits,
                **selector_aux,
                **causal_aux,
                **world_aux,
            }

        if q_family_id is not None:
            if not isinstance(q_family_id, torch.Tensor):
                q_family_id = torch.tensor(q_family_id, dtype=torch.long, device=logits.device)
            q_family_id = q_family_id.to(logits.device).long().view(-1)
            k_feat = self._normalize_knowledge_feat(knowledge_feat, cand_feat)
            knowledge_score = self.score_knowledge_support(cand_feat, mem_pool, k_feat, q_family_id)
            aux["knowledge_score"] = knowledge_score
            aux["fused_score"] = answer_score + self.lambda_knowledge * knowledge_score

        if return_aux:
            return aux
        return logits
        

    def forward_text(self, text_queries, device, has_ans=False):
        """
        text_queries : list of question str 
        out: text_embedding: bs, len, dim
            mask: bs, len (bool) [1,1,1,1,0,0]
        """
        tokenized_queries = self.tokenizer(text_queries, padding='longest', return_tensors='pt')
        # tokenized_queries = self.tokenizer(text_queries, padding='max_length', 
        #                                     max_length=self.qa_max_len if has_ans else self.q_max_len, 
        #                                     return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        
        # Use no_grad when freezing (inference_mode causes issues with backprop)
        if self.freeze_text_encoder:
            with torch.no_grad():
                encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
            # Detach and clone to allow gradient flow through text_proj
            encoded_text = encoded_text.detach().clone()
        else:
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
        
        # Project text from 768 to d_model
        encoded_text = self.text_proj(encoded_text)

        return encoded_text, tokenized_queries.attention_mask.bool()
    


class BBoxPosEmbed2D(nn.Module):
    """2D sinusoidal positional embedding for normalized bboxes [x1,y1,x2,y2] in [0,1].

    Splits the bbox into 4 components (cx, cy, w, h), each encoded with d_model/4
    sin/cos features, then concatenated into a d_model-vector. This recovers
    spatial information that gets erased when bbox is concat'd with high-dim
    semantic features and passed through a single LayerNorm.
    """

    def __init__(self, d_model, dropout=0.0, temperature=10000):
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError(f"d_model must be divisible by 8 for BBoxPosEmbed2D (4 components x even num_pos_feats); got {d_model}")
        self.d_model = d_model
        self.num_pos_feats = d_model // 4
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bbox):
        # bbox: [..., 4] in [0, 1] order [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox.unbind(-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        coords = torch.stack([cx, cy, w, h], dim=-1) * (2.0 * math.pi)  # [..., 4]

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=bbox.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos = coords[..., None] / dim_t  # [..., 4, num_pos_feats]
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
        # pos: [..., 4, num_pos_feats]
        pos = pos.flatten(-2)  # [..., d_model]
        return self.dropout(pos)


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
