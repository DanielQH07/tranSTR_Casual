from builtins import print, tuple
from signal import pause
import torch
import torch.nn as nn
import json
import re
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
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
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

class VideoQAmodel(nn.Module):
    def __init__(
        self,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        n_query=5,
        objs=20,
        frames=16,
        topK_frame=4,
        topK_obj=5,
        hard_eval=False,
        frame_feat_dim=4096,
        obj_feat_dim=2053,
        use_grounding_dino=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=None,
        minilm_name="sentence-transformers/all-MiniLM-L6-v2",
        freeze_minilm=True,
        qwen_name="Qwen/Qwen2.5-VL-3B-Instruct",
        use_qwen_lora=True,
        qwen_lora_r=8,
        qwen_lora_alpha=16,
        qwen_lora_dropout=0.1,
        qwen_lora_target_modules=None,
        evidence_top_m=5,
        use_qwen_causal_filter=False,
        qwen_max_new_tokens=64,
        **kwargs,
    ):
        super(VideoQAmodel, self).__init__()
        self.d_model = kwargs['d_model']
        encoder_dropout = kwargs['encoder_dropout']
        self.mc = n_query
        self.hard_eval = hard_eval
        self.use_grounding_dino = use_grounding_dino
        self.objs = objs
        # MiniLM for question-guided frame selection
        self.minilm = AutoModel.from_pretrained(minilm_name)
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(minilm_name)
        self.freeze_minilm = freeze_minilm
        if freeze_minilm:
            for p in self.minilm.parameters():
                p.requires_grad_(False)
        minilm_hidden = getattr(self.minilm.config, "hidden_size", 384)
        self.minilm_proj = nn.Linear(minilm_hidden, self.d_model)
        # Optional cached text projection (legacy cached features are 768-d)
        self.text_proj = nn.Linear(768, self.d_model)

        # Qwen for causal filtering + answer prediction
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_name)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        self.use_qwen_lora = use_qwen_lora
        if self.use_qwen_lora:
            if get_peft_model is None:
                raise ImportError(
                    "PEFT is required for Qwen LoRA. Please install it with: pip install peft"
                )
            if qwen_lora_target_modules is None:
                qwen_lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif isinstance(qwen_lora_target_modules, str):
                qwen_lora_target_modules = [m.strip() for m in qwen_lora_target_modules.split(",") if m.strip()]

            qwen_lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=qwen_lora_r,
                lora_alpha=qwen_lora_alpha,
                lora_dropout=qwen_lora_dropout,
                target_modules=qwen_lora_target_modules,
                bias="none",
            )
            self.qwen = get_peft_model(self.qwen, qwen_lora_cfg)

        self.qwen_hidden = getattr(self.qwen.config, "hidden_size", self.d_model)
        self.evidence_projector = nn.Linear(self.d_model, self.qwen_hidden)
        self.evidence_top_m = evidence_top_m
        self.use_qwen_causal_filter = use_qwen_causal_filter
        self.qwen_max_new_tokens = qwen_max_new_tokens
        self._init_qwen_digit_ids()

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
        
        # Evidence scoring for frame tokens
        self.mem_evidence_head = nn.Linear(self.d_model, 1)
        self.frame_sorter = PerturbedTopK(self.frame_topK)
        
        # obj_sorter only if NOT using GroundingDINO (already filtered by text prompt)
        if not use_grounding_dino:
            self.obj_sorter = PerturbedTopK(self.obj_topK)

        # hierarchy 1: obj & frame
        self.obj_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.frame_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.fo_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        
        self.vl_encoder = TransformerEncoder(TransformerEncoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))
        self.ans_decoder = TransformerDecoder(TransformerDecoderLayer(**kwargs), kwargs['num_encoder_layers'],norm=nn.LayerNorm(self.d_model))

        # position embedding
        self.pos_encoder_1d = PositionEmbeddingSine1D()

        # Generalized scoring heads (legacy)
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

    def _init_qwen_digit_ids(self):
        digit_ids = []
        for digit in ["0", "1", "2", "3", "4"]:
            tokens = self.qwen_tokenizer(digit, add_special_tokens=False).input_ids
            if not tokens:
                raise ValueError("Qwen tokenizer produced empty tokens for digit.")
            digit_ids.append(tokens[0])
        self.qwen_digit_token_ids = torch.tensor(digit_ids, dtype=torch.long)

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

    def _encode_question_minilm(self, text_queries, device):
        tokenized = self.minilm_tokenizer(text_queries, padding="longest", return_tensors="pt")
        tokenized = tokenized.to(device)
        if self.freeze_minilm:
            with torch.no_grad():
                encoded = self.minilm(**tokenized).last_hidden_state
            encoded = encoded.detach().clone()
        else:
            encoded = self.minilm(**tokenized).last_hidden_state
        encoded = self.minilm_proj(encoded)
        return encoded, tokenized.attention_mask.bool()

    def _split_answers(self, ans_word_batch):
        answers = []
        for row in ans_word_batch:
            row_answers = []
            for cand in row:
                if "[SEP]" in cand:
                    row_answers.append(cand.split("[SEP]", 1)[1].strip())
                else:
                    row_answers.append(cand.strip())
            while len(row_answers) < self.mc:
                row_answers.append("")
            answers.append(row_answers[: self.mc])
        return answers

    def _build_qwen_prompt(self, question, answers):
        lines = [
            "You are a video QA assistant.",
            "Choose the correct option index (0-4) based on the question and evidence.",
            f"Question: {question}",
            "Options:",
        ]
        for i, a in enumerate(answers):
            lines.append(f"{i}. {a}")
        lines.append("Answer with a single digit: 0, 1, 2, 3, or 4.")
        return "\n".join(lines)

    def _build_causal_filter_prompt(self, question, evidence_indices):
        lines = [
            "You are a causal evidence selector.",
            "Return a JSON object with key important_frames as a list of indices.",
            "Evidence frames:",
        ]
        for idx in evidence_indices:
            lines.append(f"[frame {idx}]")
        lines.append(f"Question: {question}")
        lines.append("Which evidence frames are causally relevant?")
        return "\n".join(lines)

    def _parse_causal_filter_output(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                payload = json.loads(text[start : end + 1])
                frames = payload.get("important_frames", [])
                return [int(x) for x in frames if str(x).isdigit()]
        except Exception:
            pass

        matches = re.findall(r"\d+", text)
        return [int(x) for x in matches]

    def _qwen_causal_filter(self, questions, evidence_indices_batch):
        device = self.qwen.device
        prompts = []
        for question, evidence_indices in zip(questions, evidence_indices_batch):
            prompts.append(self._build_causal_filter_prompt(question, evidence_indices))

        tokenized = self.qwen_tokenizer(prompts, padding=True, return_tensors="pt")
        tokenized = tokenized.to(device)
        outputs = self.qwen.generate(
            **tokenized,
            max_new_tokens=self.qwen_max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        decoded = self.qwen_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        filtered_indices = []
        for evidence_indices, text in zip(evidence_indices_batch, decoded):
            parsed = self._parse_causal_filter_output(text)
            if not parsed:
                filtered_indices.append(evidence_indices)
                continue
            filtered = [idx for idx in evidence_indices if idx in parsed]
            filtered_indices.append(filtered if filtered else evidence_indices)
        return filtered_indices

    def _qwen_forward(self, evidence_tokens, questions, answers, target_ids=None):
        device = evidence_tokens.device
        prompts = [self._build_qwen_prompt(q, a) for q, a in zip(questions, answers)]
        tokenized = self.qwen_tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        prompt_embeds = self.qwen.get_input_embeddings()(input_ids)
        evidence_embeds = self.evidence_projector(evidence_tokens)
        inputs_embeds = torch.cat([evidence_embeds, prompt_embeds], dim=1)

        evidence_mask = torch.ones(
            (evidence_tokens.size(0), evidence_tokens.size(1)),
            dtype=attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([evidence_mask, attention_mask], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :]
        digit_ids = self.qwen_digit_token_ids.to(device)
        digit_logits = logits.index_select(-1, digit_ids)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(digit_logits, target_ids)
        return digit_logits, loss

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

    def forward(self, frame_feat, obj_feat, qns_word, ans_word, return_aux=False, q_family_id=None, knowledge_feat=None, ans_id=None):
        """
        :param frame_feat:[bs, T, frame_feat_dim] e.g., [bs, 16, 4096]
        :param obj_feat:[bs, T, O, obj_feat_dim] e.g., [bs, 16, 20, 2053]
        :param qns_word: ('what are three people sitting on?', 'what is a family having?')
        :param ans_word: list of answer candidates per sample
        :param ans_id: [bs] target answer indices (0-4)
        :return:
        """
        # Size
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        # Resize frame features to d_model
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # encode q via MiniLM
        q_local, q_mask = self._encode_question_minilm(list(qns_word), device)


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
        obj_local = self.obj_resize(obj_feat)
        
        # Repeat q_local and q_mask for each frame (handle potential batch size mismatch)
        q_local_repeated = q_local.repeat_interleave(self.frame_topK, dim=0)
        q_mask_repeated = q_mask.repeat_interleave(self.frame_topK, dim=0) if q_mask is not None else None
        
        obj_local, obj_att = self.obj_decoder(obj_local.flatten(0,1),
                                            q_local_repeated, 
                                            memory_key_padding_mask=q_mask_repeated,
                                            output_attentions=True
                                            )  # b*16,O,d        #.view(B, F, O, -1) # b,16,O,d

        # GroundingDINO: skip obj_sorter (already filtered by text prompt)
        if self.use_grounding_dino:
            # Use all objects (already relevant from text-prompted detection)
            obj_local = obj_local.view(B, self.frame_topK, O, -1)  # [B, frame_topK, objs, d_model]
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
        
        # evidence scoring over frame tokens
        frame_tokens = mem[:, : self.frame_topK, :]
        evidence_scores = self.mem_evidence_head(frame_tokens).squeeze(-1)
        top_m = min(self.evidence_top_m, frame_tokens.size(1))
        top_idx = torch.topk(evidence_scores, k=top_m, dim=1).indices
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, frame_tokens.size(-1))
        evidence_tokens = frame_tokens.gather(1, idx_expand)

        if self.use_qwen_causal_filter and not self.training:
            evidence_indices_batch = top_idx.detach().cpu().tolist()
            filtered_indices = self._qwen_causal_filter(list(qns_word), evidence_indices_batch)
            max_keep = max(len(x) for x in filtered_indices)
            max_keep = max(max_keep, 1)
            filtered_tokens = []
            filtered_idx_tensor = []
            for row_idx, keep in enumerate(filtered_indices):
                keep = keep[:max_keep]
                if len(keep) < max_keep:
                    keep = keep + keep[: max_keep - len(keep)]
                keep_tensor = torch.tensor(keep, device=device, dtype=top_idx.dtype)
                filtered_idx_tensor.append(keep_tensor)
                token_idx = keep_tensor.unsqueeze(-1).expand(-1, frame_tokens.size(-1))
                filtered_tokens.append(frame_tokens[row_idx].gather(0, token_idx))
            top_idx = torch.stack(filtered_idx_tensor, dim=0)
            evidence_tokens = torch.stack(filtered_tokens, dim=0)

        answers = self._split_answers(ans_word)
        target_ids = None
        if ans_id is not None:
            target_ids = ans_id.to(device)
        qwen_logits, qwen_loss = self._qwen_forward(
            evidence_tokens,
            list(qns_word),
            answers,
            target_ids=target_ids,
        )

        mem_pool = self.pool_memory(mem, mem_mask=frame_qns_mask)
        aux = {
            "mem": mem,
            "mem_pool": mem_pool,
            "evidence_scores": evidence_scores,
            "evidence_top_idx": top_idx,
            "evidence_tokens": evidence_tokens,
            "qwen_logits": qwen_logits,
            "qwen_loss": qwen_loss,
        }

        if return_aux:
            return aux
        return qwen_logits
    
    def forward_cached(self, frame_feat, obj_feat, text_feat, return_aux=False, q_family_id=None, knowledge_feat=None, ans_id=None):
        """
        Forward pass using pre-extracted text features (bypasses DeBERTa).
        
        :param frame_feat: [bs, T, frame_feat_dim]
        :param obj_feat: [bs, T, O, obj_feat_dim]
        :param text_feat: [bs, 5, 768] - pre-extracted [CLS] features for each QA pair
        """
        B, F, O = obj_feat.size()[:3]
        device = frame_feat.device
        
        # Resize frame features
        frame_feat = self.frame_resize(frame_feat)  # [B, F, d_model]
        
        # Project cached text features (768 -> d_model)
        text_feat_proj = self.text_proj(text_feat)  # [B, 5, d_model]
        
        # Use MEAN of 5 choices as question representation for video attention
        # This approximates the question since all choices share the same question
        q_local = text_feat_proj.mean(dim=1, keepdim=True)  # [B, 1, d_model]
        q_mask = torch.zeros(B, 1, device=device).bool()
        
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
        
        # Answer decoding - use individual QA features as answer queries
        # text_feat_proj is [B, 5, d_model] - each is [CLS] of "question + answer_i"
        tgt = text_feat_proj  # [B, 5, d_model]
        out = self.ans_decoder(tgt, mem, memory_key_padding_mask=frame_qns_mask)
        
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
                "verifier_logits": verifier_logits
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
        return self._encode_question_minilm(text_queries, device)
    


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