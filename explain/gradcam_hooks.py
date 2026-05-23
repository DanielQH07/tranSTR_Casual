"""Grad-CAM + attention rollout for the TranSTR multimodal transformer.

Hooks tap into 4 intermediate tensors:
    - frame_local  : output of `model.frame_decoder` (frame-level features after Q-attended)
    - obj_local    : output of `model.obj_decoder`   (object-level features after Q-attended)
    - frame_obj    : output of `model.fo_decoder`    (frame×obj fused features)
    - mem          : output of `model.vl_encoder`    (unified memory [frame_obj | q_tokens])

For Grad-CAM we additionally need q_local (post text_proj). Because text encoding lives
inside `model.forward_text`, we expose a thin custom forward that mirrors `model.forward`
but keeps the autograd graph alive (no `@torch.no_grad`) and stores hookable references.
"""
from __future__ import annotations

from itertools import chain
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from networks.topk import HardtopK


# ---------------------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------------------


def _gradcam_from_tensor(activation: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Compute Grad-CAM heatmap from `[B, N, d]` activation/grad pair.

    Returns `[B, N]` non-negative, max-normalized per batch sample.
    """
    if activation.dim() != grad.dim():
        raise ValueError(
            f"activation/grad rank mismatch: {activation.shape} vs {grad.shape}"
        )

    # Channel-pool gradients to get per-token weight (alpha)
    alpha = grad.mean(dim=-1, keepdim=True)  # [B, N, 1]
    cam = (alpha * activation).sum(dim=-1)   # [B, N]
    cam = F.relu(cam)
    # Per-sample max-normalize for plotting stability
    cam_max = cam.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    cam = cam / cam_max
    return cam


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------


def _rollout_attention(att_list: List[torch.Tensor]) -> torch.Tensor:
    """Naive attention rollout (Abnar & Zuidema, 2020) for self-attention chains.

    Each `att` in `att_list` should be `[B, N, N]` (already head-averaged).
    Returns the rolled-out `[B, N, N]` matrix.
    """
    rolled = None
    for att in att_list:
        # Add identity to model residual connection, renormalize
        eye = torch.eye(att.size(-1), device=att.device).unsqueeze(0)
        a = att + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rolled = a if rolled is None else torch.bmm(a, rolled)
    return rolled


# ---------------------------------------------------------------------------
# MultiTargetGradCAM
# ---------------------------------------------------------------------------


class MultiTargetGradCAM:
    """Run a hook-aware forward + backward and return per-modality Grad-CAM maps.

    Usage:
        cam = MultiTargetGradCAM(model)
        out = cam.run(frame_feat, obj_feat, qns_word, ans_word, q_family_id, target_class=None)
        # out["frame_cam"], out["obj_cam"], out["q_cam"], out["mem_cam"], ...
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _select_topk_indices(sorter, att_map: torch.Tensor, n_items: int, hard_eval: bool, training: bool) -> torch.Tensor:
        """Mirror model.forward top-k routing for both frame/object branches."""
        if training:
            return rearrange(
                sorter(att_map.flatten(1, 2)),
                "b (n q) k -> b n q k",
                n=n_items,
            ).sum(-2)

        if hard_eval:
            return rearrange(
                HardtopK(att_map.flatten(1, 2), sorter.k),
                "b (n q) k -> b n q k",
                n=n_items,
            ).sum(-2)

        return rearrange(
            sorter(att_map.flatten(1, 2)),
            "b (n q) k -> b n q k",
            n=n_items,
        ).sum(-2)

    # ------------------------------------------------------------------
    # Custom hookable forward (mirrors model.forward but keeps grads)
    # ------------------------------------------------------------------
    def _forward_hooked(
        self,
        frame_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        qns_word,
        ans_word,
        q_family_id: Optional[torch.Tensor] = None,
        knowledge_feat=None,
    ) -> Dict[str, torch.Tensor]:
        model = self.model
        device = frame_feat.device
        B, F_total, O = obj_feat.size()[:3]

        frame_feat_proj = model.frame_resize(frame_feat)

        q_local, q_mask = model.forward_text(list(qns_word), device)
        # Detach-clone path (when frozen) breaks grad — re-enable leaf grad on q_local
        if not q_local.requires_grad:
            q_local = q_local.detach().clone().requires_grad_(True)
        q_local.retain_grad()

        # ---- frame decoder ----
        frame_mask = torch.ones(B, F_total, dtype=torch.bool, device=device)
        frame_local, frame_att = model.frame_decoder(
            frame_feat_proj,
            q_local,
            memory_key_padding_mask=q_mask,
            query_pos=model.pos_encoder_1d(frame_mask, model.d_model),
            output_attentions=True,
        )
        frame_local.retain_grad()

        # frame top-K selection (same logic as model.forward train/eval/hard_eval branch)
        idx_frame = self._select_topk_indices(
            sorter=model.frame_sorter,
            att_map=frame_att,
            n_items=F_total,
            hard_eval=model.hard_eval,
            training=model.training,
        )
        frame_local_top = (frame_local.transpose(1, 2) @ idx_frame).transpose(1, 2)

        # ---- object decoder ----
        # Keep object path parity with model.forward:
        # 1) select objects by selected frames (hard gather if configured)
        # 2) fit object width
        # 3) encode via semantic+bbox path when enabled
        obj_feat_top = model._select_obj_by_frame(obj_feat, idx_frame)
        obj_feat_top = model._fit_obj_feat_dim(obj_feat_top)
        obj_local_in = model._encode_objects(obj_feat_top)

        q_local_rep = q_local.repeat_interleave(model.frame_topK, dim=0)
        q_mask_rep = q_mask.repeat_interleave(model.frame_topK, dim=0)
        obj_local, obj_att = model.obj_decoder(
            obj_local_in.flatten(0, 1),
            q_local_rep,
            memory_key_padding_mask=q_mask_rep,
            output_attentions=True,
        )
        obj_local.retain_grad()

        if model.use_grounding_dino:
            obj_local_view = obj_local.view(B, model.frame_topK, O, -1)
        else:
            idx_obj = self._select_topk_indices(
                sorter=model.obj_sorter,
                att_map=obj_att,
                n_items=O,
                hard_eval=model.hard_eval,
                training=model.training,
            )
            obj_local_view = (
                obj_local.transpose(1, 2) @ idx_obj
            ).transpose(1, 2).view(B, model.frame_topK, model.obj_topK, -1)

        # ---- frame-object fusion ----
        frame_obj, fo_att = model.fo_decoder(
            frame_local_top,
            obj_local_view.flatten(1, 2),
            output_attentions=True,
        )
        frame_obj = frame_obj.view(B, model.frame_topK, -1)
        frame_obj.retain_grad()

        # ---- unified memory ----
        frame_mask_top = torch.ones(B, model.frame_topK, dtype=torch.bool, device=device)
        frame_qns_mask = torch.cat((frame_mask_top, q_mask), dim=1).bool()
        mem_in = torch.cat((frame_obj, q_local), dim=1)
        mem = model.vl_encoder(
            mem_in,
            src_key_padding_mask=frame_qns_mask,
            pos=model.pos_encoder_1d(frame_qns_mask.bool(), model.d_model),
        )
        mem.retain_grad()

        # ---- answer decoder ----
        a_seq, _ = model.forward_text(list(chain(*ans_word)), device, has_ans=True)
        a_seq = rearrange(a_seq, "(n b) t c -> b n t c", b=B)
        tgt = a_seq[:, :, 0, :]
        out, ans_att = model.ans_decoder(
            tgt,
            mem,
            memory_key_padding_mask=frame_qns_mask,
            output_attentions=True,
        )
        cand_feat, answer_score, evidence_score = model.decode_candidates(out)
        mem_pool = model.pool_memory(mem, mem_mask=frame_qns_mask)

        fused_score = answer_score
        knowledge_score = None
        if q_family_id is not None:
            if not isinstance(q_family_id, torch.Tensor):
                q_family_id = torch.tensor(
                    q_family_id, dtype=torch.long, device=device
                )
            q_family_id = q_family_id.to(device).long().view(-1)
            k_feat = model._normalize_knowledge_feat(knowledge_feat, cand_feat)
            knowledge_score = model.score_knowledge_support(
                cand_feat, mem_pool, k_feat, q_family_id
            )
            fused_score = answer_score + model.lambda_knowledge * knowledge_score

        return {
            "fused_score": fused_score,
            "answer_score": answer_score,
            "evidence_score": evidence_score,
            "knowledge_score": knowledge_score,
            # tensors to compute Grad-CAM on
            "q_local": q_local,
            "frame_local": frame_local,
            "obj_local": obj_local,
            "frame_obj": frame_obj,  # Note: this is the post-view tensor; its grad is on the pre-view buffer above
            "mem": mem,
            # raw forward attentions (head-averaged inside MultiheadAttention)
            "frame_att": frame_att,
            "obj_att": obj_att,
            "fo_att": fo_att,
            "ans_att": ans_att,
            # bookkeeping
            "frame_qns_mask": frame_qns_mask,
            "topK": model.frame_topK,
            "F_total": F_total,
            "O": O,
            "q_len": q_local.size(1),
            "idx_frame": idx_frame,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        frame_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        qns_word,
        ans_word,
        device: torch.device,
        q_family_id: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        per_class: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run forward+backward, return Grad-CAM maps for the predicted (or chosen) class.

        Args:
            target_class: if None, use argmax of `fused_score`.
            per_class:   if True, also compute Grad-CAM for every candidate class.
        """
        model = self.model
        was_training = model.training
        model.eval()

        # Need autograd ON even in eval
        with torch.set_grad_enabled(True):
            fwd = self._forward_hooked(
                frame_feat, obj_feat, qns_word, ans_word, q_family_id
            )

            fused = fwd["fused_score"]  # [B, N_cands]
            probs = torch.softmax(fused, dim=-1)
            pred = int(fused.argmax(dim=-1)[0].item())
            tgt_cls = int(target_class) if target_class is not None else pred

            # Backward to populate `.grad` on retained tensors
            model.zero_grad(set_to_none=True)
            score = fused[0, tgt_cls]
            score.backward(retain_graph=per_class)

            cams = self._collect_cams(fwd)

            per_class_cams = None
            if per_class:
                per_class_cams = {}
                for cls in range(fused.size(1)):
                    if cls == tgt_cls:
                        per_class_cams[cls] = cams
                        continue
                    model.zero_grad(set_to_none=True)
                    # zero retained grads to avoid accumulation
                    for k in ("q_local", "frame_local", "obj_local", "frame_obj", "mem"):
                        t = fwd.get(k)
                        if isinstance(t, torch.Tensor) and t.grad is not None:
                            t.grad = None
                    fused[0, cls].backward(retain_graph=(cls != fused.size(1) - 1))
                    per_class_cams[cls] = self._collect_cams(fwd)

            # Attention rollout (forward only)
            with torch.no_grad():
                rollout = self._build_rollout(fwd, tgt_cls)

        if was_training:
            model.train()

        out = {
            "pred": pred,
            "target_class": tgt_cls,
            "probs": probs[0].detach().cpu().numpy(),
            "fused_score": fwd["fused_score"][0].detach().cpu().numpy(),
            "answer_score": fwd["answer_score"][0].detach().cpu().numpy(),
            "evidence_score": fwd["evidence_score"][0].detach().cpu().numpy(),
            "knowledge_score": (
                fwd["knowledge_score"][0].detach().cpu().numpy()
                if fwd["knowledge_score"] is not None
                else None
            ),
            # Raw forward attentions (for compare plot)
            "frame_att": fwd["frame_att"][0].detach().cpu().numpy(),
            "obj_att": fwd["obj_att"].detach().cpu().numpy(),  # [B*topK, O, q_len]
            "ans_att": fwd["ans_att"][0].detach().cpu().numpy(),
            # Selected frame indices via frame_sorter
            "selected_frame_indices": (
                fwd["idx_frame"][0].argmax(dim=0).detach().cpu().numpy()
            ),
            # CAM maps
            "cams": cams,
            "per_class_cams": per_class_cams,
            "rollout_frame": rollout,
            # bookkeeping
            "topK": fwd["topK"],
            "F_total": fwd["F_total"],
            "q_len": fwd["q_len"],
        }
        return out

    # ------------------------------------------------------------------
    # CAM collection / rollout
    # ------------------------------------------------------------------
    def _collect_cams(self, fwd: Dict[str, torch.Tensor]) -> Dict[str, "torch.Tensor"]:
        cams = {}

        def _safe(name):
            t = fwd[name]
            if t.grad is None:
                return None
            return _gradcam_from_tensor(t.detach(), t.grad.detach())

        cams["q_cam"] = _safe("q_local")  # [B, q_len]

        # frame_cam: cam over frame_local pre-topK -> [B, F_total]
        cams["frame_cam"] = _safe("frame_local")

        # obj_cam: obj_local has shape [B*topK, O, d] -> reshape to [B, topK, O]
        obj_cam = _safe("obj_local")
        topK = fwd["topK"]
        if obj_cam is not None:
            B = fwd["frame_local"].size(0)
            cams["obj_cam"] = obj_cam.view(B, topK, fwd["O"])
        else:
            cams["obj_cam"] = None

        # frame_obj cam over post-fusion frame slots -> [B, topK]
        cams["frame_obj_cam"] = _safe("frame_obj")

        # mem cam over [B, topK + q_len] -> split visual vs question
        mem_cam = _safe("mem")
        if mem_cam is not None:
            cams["mem_cam"] = mem_cam
            cams["mem_visual_cam"] = mem_cam[:, :topK]
            cams["mem_question_cam"] = mem_cam[:, topK:]
        else:
            cams["mem_cam"] = None
            cams["mem_visual_cam"] = None
            cams["mem_question_cam"] = None

        # Convert to numpy for downstream plotting
        for k, v in list(cams.items()):
            if isinstance(v, torch.Tensor):
                cams[k] = v.detach().cpu().numpy()
        return cams

    def _build_rollout(
        self, fwd: Dict[str, torch.Tensor], tgt_cls: int
    ) -> Optional["torch.Tensor"]:
        """End-to-end attention rollout for the target answer class onto frame timeline.

        Combines: ans_decoder cross-attn (ans_q -> mem)
                   × vl_encoder (no rollout; treat as identity for stability)
                   × frame_decoder cross-attn (frame -> q)

        Returns a 1-D numpy array of shape [F_total] for the predicted class.
        """
        try:
            ans_att = fwd["ans_att"]      # [B, N_ans, mem_len]
            frame_att = fwd["frame_att"]  # [B, F_total, q_len]
            topK = fwd["topK"]

            # Pull mem-position weights for predicted answer
            mem_w = ans_att[0, tgt_cls]   # [mem_len]
            visual_w = mem_w[:topK]       # weight on each top-K slot
            question_w = mem_w[topK:]     # weight on each q token

            # Map top-K slots -> original F_total via idx_frame (soft selection mass)
            idx_frame = fwd["idx_frame"][0]  # [F_total, topK]
            frame_w_from_visual = (idx_frame * visual_w.unsqueeze(0)).sum(dim=-1)  # [F_total]

            # Map question weight -> frames via frame_decoder cross-attn
            # frame_att[0]: [F_total, q_len]
            frame_w_from_q = (frame_att[0] * question_w.unsqueeze(0)).sum(dim=-1)

            rollout = frame_w_from_visual + frame_w_from_q
            rollout = F.relu(rollout)
            rollout = rollout / rollout.max().clamp(min=1e-8)
            return rollout.detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            print(f"[rollout] skipped: {exc}")
            return None


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_gradcam_inference(
    model,
    frame_feat: torch.Tensor,
    obj_feat: torch.Tensor,
    qns_word,
    ans_word,
    device: torch.device,
    q_family_id: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    per_class: bool = False,
):
    """One-shot helper: instantiate `MultiTargetGradCAM` and call `.run`."""
    cam = MultiTargetGradCAM(model)
    return cam.run(
        frame_feat,
        obj_feat,
        qns_word,
        ans_word,
        device=device,
        q_family_id=q_family_id,
        target_class=target_class,
        per_class=per_class,
    )
