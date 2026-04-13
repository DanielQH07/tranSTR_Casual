# TranSTR-Causal Input Guide (DINOv3 + Grounded Version)

This document defines the updated input format and implementation notes for your modified pipeline:
- ViT is replaced by DINOv3 frame features.
- Object features are produced by Grounded.
- Object selection top-K inside the model is removed.
- Grounding is question-driven, and detections can be precomputed for all 6 question types per video.

The goal is to keep tensor shapes stable and avoid frame-index or box-coordinate mismatch when refactoring the code.

## 1. Updated Data Layout

Recommended structure:

```text
/data_root/
    ├── annotations/
    │     ├── train.csv
    │     ├── val.csv
    │     └── test.csv
    │
    ├── frame_feat_dinov3/
    │     ├── train/
    │     │     ├── <video_id>.pt
    │     │     └── ...
    │     ├── val/
    │     └── test/
    │
    └── grounded_objects/
                ├── train/
                │     ├── <video_id>_0.pkl   # descriptive
                │     ├── <video_id>_1.pkl   # explanatory
                │     ├── <video_id>_2.pkl   # predictive answer
                │     ├── <video_id>_3.pkl   # predictive reason
                │     ├── <video_id>_4.pkl   # counterfactual answer
                │     └── <video_id>_5.pkl   # counterfactual reason
                ├── val/
                └── test/
```

Why this key format matters:
- The same video can produce different grounded objects for different question types.
- Use `video_id + qtype` as the object-feature key, not only `video_id`.

## 2. Annotation Requirements

CSV columns should include:
- `video_id`
- `question`
- `answer` (correct option index or text, according to your loader logic)
- `a0` to `a4`
- `type` (or any field mappable to qtype 0..5)

Question type convention (recommended):
- `0`: descriptive
- `1`: explanatory
- `2`: predictive answer
- `3`: predictive reason
- `4`: counterfactual answer
- `5`: counterfactual reason

## 3. DINOv3 Frame Features

You still sample exactly 16 frames per video (same policy as old ViT pipeline). Only the extractor is changed.

### Option A: Global frame features

- Per video tensor: `[16, D_frame]`
- Batch tensor: `[B, 16, D_frame]`

Use this if you do not need true ROI pooling from feature maps.

### Option B: Spatial frame maps (recommended for ROI feature extraction)

- Per video tensor: `[16, C, Hf, Wf]`
- Batch tensor: `[B, 16, C, Hf, Wf]`

Use this if you want ROIAlign/grid sampling for each grounded box.

Important:
- If you only store global vectors, you cannot perform real ROI feature cropping.

## 4. Grounded-SAM Object Package

For each `(video_id, qtype)` file, store per-frame object results for all 16 sampled frames.

Minimum fields per frame:
- `boxes_xyxy_norm`: `[N, 4]` in normalized coordinates `[0,1]`
- `scores`: `[N]`
- `labels`: `[N]` (label id or text id)
- `valid_mask`: `[N]` (1 valid, 0 padded)

After padding to `N_max` objects per frame:
- `boxes`: `[16, N_max, 4]`
- `scores`: `[16, N_max]`
- `labels`: `[16, N_max]`
- `obj_mask`: `[16, N_max]`

Batch form:
- `boxes`: `[B, 16, N_max, 4]`
- `scores`: `[B, 16, N_max]`
- `labels`: `[B, 16, N_max]`
- `obj_mask`: `[B, 16, N_max]`

## 5. Coordinate and Resize Consistency (Critical)

You removed old geometric harmonization, but you still must keep one strict coordinate convention.

Required metadata to store with detections:
- original image size: `orig_h`, `orig_w`
- detector input size: `det_h`, `det_w`
- resize mode: `stretch` or `letterbox`
- `scale`, `pad_x`, `pad_y` if letterbox is used

Rule:
- The coordinate system used to save boxes must match the coordinate system used when ROI features are extracted.
- If detector preprocessing and encoder preprocessing differ, implement an explicit box transform before ROI pooling.

## 6. New Model-Side Contract

You keep frame selection, but remove object top-K selection.

### 6.1 What stays

- Text encoder (`forward_text`)
- Frame decoder + frame top-K selection
- Frame-object fusion (`fo_decoder`)
- Vision-language encoder (`vl_encoder`)
- Answer decoder and classifier

### 6.2 What changes

- Remove `obj_sorter` and object top-K routing.
- Build object tokens directly from grounded objects.
- Optionally add ROI feature projection from frame maps.

Suggested object token composition:

```text
obj_token = LN(
        W_roi(roi_feat)
    + W_box(box_positional_encoding)
    + W_lbl(label_embedding)
    + W_scr(score_embedding)
)
```

Output shape for object branch:
- `obj_token`: `[B, Kf, N_max, d_model]`

Then flatten for frame-object fusion memory:
- `obj_token_flat`: `[B, Kf * N_max, d_model]`

## 7. DataLoader Refactor Checklist

Update `__getitem__` to return:
1. DINOv3 frame features
2. Grounded-SAM object package (`boxes`, `scores`, `labels`, `obj_mask`)
3. question text
4. 5 answer candidates
5. answer id
6. question key (`video_id_qtype`)

Enforce assertions before returning:
- frame count is always 16
- object tensors are always padded to `N_max`
- object keys exist for each `(video_id, qtype)` sample

## 8. Forward-Pass Shape Checks

Before full training, validate these tensors:
- `q_local`: `[B, Lq, d_model]`
- `frame_local_raw`: `[B, Kf, d_model]`
- `obj_token`: `[B, Kf, N_max, d_model]`
- `frame_obj`: `[B, Kf, d_model]`
- `mem`: `[B, Kf + Lq, d_model]`
- `logits`: `[B, 5]`

If ROI extraction is enabled:
- `roi_feat`: `[B, Kf, N_max, C_roi]`

## 9. Practical Migration Plan

Step 1: Finalize feature format
- Decide DINOv3 output type: global or spatial map.
- Fix `N_max` objects per frame.
- Freeze one coordinate convention.

Step 2: Refactor DataLoader
- Load DINOv3 features.
- Load grounded object package by `video_id_qtype`.
- Pad objects and create masks.

Step 3: Refactor model object branch
- Remove object top-K components.
- Add grounded object encoder module.
- Gather objects by selected top-K frames.
- Fuse with frame branch.

Step 4: Alignment debugging
- Visualize boxes on raw frames.
- Visualize transformed boxes on encoder feature coordinates.
- Verify no frame-index drift between frame features and object files.

Step 5: Train in two phases
- Phase A: box + label + score only.
- Phase B: add ROI feature projection.
- Compare against baseline on d/e/par/car/all.

## 10. Common Failure Modes

1. Frame index mismatch
- Symptom: random training behavior, unstable metrics.
- Fix: store and assert sampled frame indices.

2. Coordinate mismatch
- Symptom: ROI features do not correspond to visible objects.
- Fix: explicit transform using saved resize metadata.

3. Too many objects per frame
- Symptom: memory spike and diluted attention.
- Fix: set `N_max`, apply confidence threshold.

4. Missing detections in some frames
- Symptom: NaN or shape inconsistency.
- Fix: zero-pad objects and rely on `obj_mask`.

## 11. Final Architecture Decision Summary

- Frame extractor: DINOv3 (16-frame sampling unchanged).
- Object extractor: Grounded-SAM, question-prompt-based.
- Object branch: direct grounded objects, no object top-K selection.
- Frame branch: keep question-guided frame top-K.
- Strong recommendation: keep spatial frame maps if ROI feature extraction is required.
