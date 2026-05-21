"""Visualization helpers for Grad-CAM outputs from `gradcam_hooks.MultiTargetGradCAM`.

All plots use the dark dashboard palette of `full-mo-inference.ipynb`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

# Palette reused from the inference notebook
_BG = "#0d1117"
_ACC = "#3fb950"
_HIGHLIGHT = "#ffd700"
_TEXT = "#c9d1d9"

_BOX_PALETTE = [
    (255, 64, 64), (64, 220, 64), (64, 160, 255), (255, 200, 32),
    (210, 96, 255), (0, 255, 220), (255, 128, 0), (255, 96, 190),
    (160, 255, 100), (120, 160, 255), (255, 255, 100), (255, 0, 128),
    (100, 255, 255), (200, 200, 200), (255, 165, 0), (138, 43, 226),
    (50, 205, 50), (255, 105, 180), (30, 144, 255), (255, 69, 0),
]


# ---------------------------------------------------------------------------
# Frame-level Grad-CAM
# ---------------------------------------------------------------------------


def plot_frame_gradcam(
    frames: List[np.ndarray],
    frame_cam: np.ndarray,
    selected_indices: Sequence[int],
    rollout: Optional[np.ndarray],
    question: str,
    qtype: str,
):
    """Show Grad-CAM heat over each frame thumbnail, with timeline bar chart."""
    n_frames = min(len(frames), len(frame_cam))
    cam = frame_cam[:n_frames]
    cam_norm = cam / (cam.max() + 1e-8)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(_BG)
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.4, wspace=0.15)

    sel_set = set(int(s) for s in selected_indices)
    for i in range(16):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        if i >= n_frames:
            ax.axis("off")
            continue
        frame = frames[i].copy()
        h, w = frame.shape[:2]
        # Red-tinted overlay proportional to cam
        overlay = np.zeros_like(frame)
        overlay[..., 0] = 255  # R
        alpha = float(cam_norm[i]) * 0.55
        blended = (frame.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha)
        blended = blended.astype(np.uint8)

        if i in sel_set:
            for spine in ax.spines.values():
                spine.set_edgecolor(_HIGHLIGHT)
                spine.set_linewidth(4)
            title_color = _HIGHLIGHT
            tag = f"F{i} ★ cam={cam[i]:.2f}"
        else:
            title_color = "#6e7681"
            tag = f"F{i}  cam={cam[i]:.2f}"
        ax.imshow(blended)
        ax.set_title(tag, fontsize=9, color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_bar = fig.add_subplot(gs[4, :])
    x = np.arange(n_frames)
    ax_bar.bar(x, cam, color="#da3633", label="Grad-CAM (frame_local)")
    if rollout is not None and len(rollout) >= n_frames:
        ax_bar.plot(x, rollout[:n_frames], color="#58a6ff", marker="o",
                    linewidth=2, label="Attention rollout")
    for s in selected_indices:
        s = int(s)
        if 0 <= s < n_frames:
            ax_bar.axvline(s, color=_HIGHLIGHT, linestyle="--", alpha=0.6)
    ax_bar.set_xlabel("Frame index")
    ax_bar.set_ylabel("Importance")
    ax_bar.set_xticks(x)
    ax_bar.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=_TEXT)
    ax_bar.set_facecolor("#161b22")
    ax_bar.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"{qtype} | Frame Grad-CAM (target = predicted answer)\nQ: {question}",
        color="#f0f6fc", fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Object Grad-CAM (bbox alpha)
# ---------------------------------------------------------------------------


def plot_object_gradcam(
    frames: List[np.ndarray],
    selected_indices: Sequence[int],
    obj_cam: np.ndarray,            # [topK, O]
    gdino_frames: list,
    gdino_sample_idx: np.ndarray,
    orig_h: int,
    orig_w: int,
    qtype: str,
    top_k: Optional[int] = None,
):
    """Draw bboxes on each selected frame; alpha & ranking from Grad-CAM."""
    n_sel = len(selected_indices)
    fig, axes = plt.subplots(n_sel, 1, figsize=(14, 8 * n_sel))
    fig.patch.set_facecolor(_BG)
    if n_sel == 1:
        axes = [axes]

    for slot, (ax, fidx) in enumerate(zip(axes, selected_indices)):
        fidx = int(fidx)
        frame = frames[fidx].copy() if 0 <= fidx < len(frames) else np.zeros((360, 640, 3), np.uint8)
        h, w = frame.shape[:2]
        gi = int(gdino_sample_idx[fidx]) if fidx < len(gdino_sample_idx) else 0
        fd = gdino_frames[gi] if gi < len(gdino_frames) else {}

        boxes = fd.get("boxes_xyxy_orig", np.zeros((0, 4), dtype=np.float32))
        labels = []
        for key in ("labels_text", "labels", "phrases", "class_names"):
            v = fd.get(key)
            if v is not None and len(v) > 0:
                labels = list(v)
                break

        scores = obj_cam[slot] if slot < len(obj_cam) else np.zeros(len(boxes))
        n_total = len(boxes)
        n_scored = int(min(n_total, len(scores)))

        sx, sy = w / orig_w, h / orig_h
        all_scores = np.zeros(n_total, dtype=np.float32)
        if n_scored > 0:
            all_scores[:n_scored] = scores[:n_scored]
        max_s = max(float(all_scores.max()) if n_total else 0.0, 1e-6)

        order = np.argsort(all_scores)[::-1]
        if top_k is not None:
            order = order[:top_k]

        box_line = max(int(min(h, w) * 0.006), 3)
        font_scale = max(min(h, w) / 900.0, 0.7)
        font_thick = max(int(font_scale * 2), 2)

        for rank, oi in enumerate(order):
            x1, y1, x2, y2 = boxes[oi]
            x1, y1 = int(x1 * sx), int(y1 * sy)
            x2, y2 = int(x2 * sx), int(y2 * sy)
            color = _BOX_PALETTE[rank % len(_BOX_PALETTE)]
            alpha = float(all_scores[oi] / max_s) if max_s > 0 else 0.0

            # Translucent fill proportional to Grad-CAM strength
            if alpha > 0.05:
                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].astype(np.float32)
                tint = np.zeros_like(roi)
                tint[..., 0] = color[0]
                tint[..., 1] = color[1]
                tint[..., 2] = color[2]
                blend = roi * (1 - 0.45 * alpha) + tint * (0.45 * alpha)
                frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = blend.astype(np.uint8)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_line)
            raw_label = labels[oi] if oi < len(labels) else f"obj{oi}"
            tag = f"#{rank + 1} {raw_label} cam={all_scores[oi]:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad = int(th * 0.35)
            bg_y1 = max(0, y1 - th - pad * 2)
            bg_y2 = bg_y1 + th + pad * 2
            cv2.rectangle(frame, (x1, bg_y1), (min(w, x1 + tw + pad * 2), bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, bg_y1), (min(w, x1 + tw + pad * 2), bg_y2), color, max(1, box_line // 2))
            cv2.putText(frame, tag, (x1 + pad, bg_y2 - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

        ax.imshow(frame)
        ax.set_title(f"Selected frame {fidx} (slot {slot + 1}) | {n_total} boxes",
                     color=_ACC, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{qtype} | Object Grad-CAM (alpha = cam strength)",
                 color=_ACC, fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Question-token Grad-CAM (HTML)
# ---------------------------------------------------------------------------


def plot_question_token_gradcam(
    tokenizer,
    question: str,
    q_cam: np.ndarray,
    qtype: str,
    answers: Optional[List[str]] = None,
    pred_idx: Optional[int] = None,
):
    """Render question tokens with background tinted by Grad-CAM intensity."""
    encoded = tokenizer(question, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    n = min(len(tokens), len(q_cam))
    cam = q_cam[:n]
    cam = cam / (cam.max() + 1e-8)

    spans = []
    for i in range(n):
        intensity = float(cam[i])
        # Map intensity to red shade
        r = int(218 * intensity + 22 * (1 - intensity))
        g = int(54 * intensity + 27 * (1 - intensity))
        b = int(51 * intensity + 34 * (1 - intensity))
        tok = tokens[i].replace("Ġ", " ").replace("▁", " ")
        spans.append(
            f"<span style='background:rgb({r},{g},{b}); color:#fff; padding:2px 4px; "
            f"margin:1px; border-radius:3px;' title='cam={cam[i]:.2f}'>{tok}</span>"
        )

    pred_html = ""
    if answers is not None and pred_idx is not None:
        pred_html = (
            f"<div style='color:#3fb950; margin-top:8px;'>Predicted: "
            f"[{pred_idx}] {answers[pred_idx]}</div>"
        )

    html = (
        f"<div style='background:#0d1117; padding:12px; border-radius:6px; "
        f"font-family:monospace;'>"
        f"<div style='color:#a371f7; font-weight:bold; margin-bottom:8px;'>"
        f"{qtype} | Question-token Grad-CAM</div>"
        f"<div>{''.join(spans)}</div>{pred_html}</div>"
    )
    display(HTML(html))


# ---------------------------------------------------------------------------
# Cross-modal mem split
# ---------------------------------------------------------------------------


def plot_mem_split_bar(
    mem_visual_cam: np.ndarray,    # [topK]
    mem_question_cam: np.ndarray,  # [q_len]
    selected_frame_indices: Sequence[int],
    qtype: str,
):
    """Bar chart: how much each mem-position contributes to the predicted answer."""
    visual = np.asarray(mem_visual_cam).ravel()
    question = np.asarray(mem_question_cam).ravel()

    total_v = float(visual.sum())
    total_q = float(question.sum())
    total = total_v + total_q + 1e-8
    pct_v = 100.0 * total_v / total
    pct_q = 100.0 * total_q / total

    fig, axes = plt.subplots(1, 2, figsize=(16, 4),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(_BG)

    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    bars = ax1.bar(["Visual\n(frame_obj)", "Question\n(tokens)"],
                   [total_v, total_q], color=["#da3633", "#58a6ff"], edgecolor="#30363d")
    for bar, pct in zip(bars, [pct_v, pct_q]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(), f"{pct:.1f}%",
                 ha="center", va="bottom", color=_TEXT, fontsize=12, fontweight="bold")
    ax1.set_title("Modality contribution (sum of Grad-CAM)",
                  color="#f0883e", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    n_v = len(visual)
    n_q = len(question)
    x = np.arange(n_v + n_q)
    colors = ["#da3633"] * n_v + ["#58a6ff"] * n_q
    ax2.bar(x, np.concatenate([visual, question]), color=colors, edgecolor="#30363d")
    ax2.axvline(n_v - 0.5, color=_HIGHLIGHT, linestyle="--", linewidth=2)
    labels = []
    for i, fi in enumerate(selected_frame_indices):
        if i < n_v:
            labels.append(f"F{int(fi)}")
    while len(labels) < n_v:
        labels.append(f"V{len(labels)}")
    labels += [f"q{i}" for i in range(n_q)]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, fontsize=8)
    ax2.set_title("Per-position Grad-CAM (red=visual slots, blue=question tokens)",
                  color="#f0883e", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(f"{qtype} | Unified-memory cross-modal Grad-CAM",
                 color="#a371f7", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Compare Attention vs Grad-CAM vs Rollout
# ---------------------------------------------------------------------------


def plot_attention_vs_gradcam(
    frame_attention_peak: np.ndarray,  # [F_total]
    frame_cam: np.ndarray,             # [F_total]
    rollout: Optional[np.ndarray],     # [F_total] or None
    selected_indices: Sequence[int],
    qtype: str,
):
    """Side-by-side comparison of three attribution methods on the frame timeline."""
    n = min(len(frame_attention_peak), len(frame_cam))
    att = frame_attention_peak[:n]
    cam = frame_cam[:n]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor("#161b22")

    x = np.arange(n)
    width = 0.4
    a_norm = att / (att.max() + 1e-8)
    c_norm = cam / (cam.max() + 1e-8)
    ax.bar(x - width / 2, a_norm, width, color="#58a6ff", label="Attention (peak)", edgecolor="#30363d")
    ax.bar(x + width / 2, c_norm, width, color="#da3633", label="Grad-CAM", edgecolor="#30363d")
    if rollout is not None and len(rollout) >= n:
        r = rollout[:n]
        r_norm = r / (r.max() + 1e-8)
        ax.plot(x, r_norm, color=_HIGHLIGHT, marker="o", linewidth=2, label="Rollout")

    for s in selected_indices:
        s = int(s)
        if 0 <= s < n:
            ax.axvline(s, color=_ACC, linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized importance")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=_TEXT)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(f"{qtype} | Attention vs Grad-CAM vs Rollout",
                 color="#a371f7", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
