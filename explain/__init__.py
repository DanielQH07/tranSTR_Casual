"""Explainability utilities for TranSTR (Grad-CAM, attention rollout)."""
from .gradcam_hooks import MultiTargetGradCAM, run_gradcam_inference
from .viz_gradcam import (
    plot_frame_gradcam,
    plot_object_gradcam,
    plot_question_token_gradcam,
    plot_mem_split_bar,
    plot_attention_vs_gradcam,
)

__all__ = [
    "MultiTargetGradCAM",
    "run_gradcam_inference",
    "plot_frame_gradcam",
    "plot_object_gradcam",
    "plot_question_token_gradcam",
    "plot_mem_split_bar",
    "plot_attention_vs_gradcam",
]
