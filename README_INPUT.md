# Causal-VidQA Input Structure Guide

This document details the exact input data structure required by the original `TranSTR/Causal-VidQA` codebase.

## 1. Directory Structure Overview
```
/data_root/
  ├── vqa/causal/anno/           <-- sample_list_path
  │     ├── train.csv
  │     ├── val.csv
  │     └── test.csv
  │
  ├── region_feat_aln/           <-- video_feature_path
  │     ├── train/
  │     │     ├── video_id_1.pt
  │     │     └── ...
  │     ├── val/
  │     └── test/
  │
  └── object_feat/               <-- object_feature_path
        ├── video_id_1/
        │     ├── frame_1.pkl
        │     ├── frame_2.pkl
        │     └── ...
        └── video_id_2/
```

---

## 2. Annotations (CSV)
**Path:** `sample_list_path/{split}.csv` (e.g., `train.csv`)

The code uses `pandas.read_csv` to load these files.

**Required Columns:**
*   `video_id`: (int/str) Unique identifier for the video.
*   `width`: (int) Width of the video frame (used for normalization).
*   `height`: (int) Height of the video frame.
*   `question`: (str) The question text.
*   `answer`: (str) The **text** of the correct answer.
*   `a0`, `a1`, `a2`, `a3`, `a4`: (str) The text for the 5 multiple-choice candidates.
*   `type`: (int/str) Question type ID (optional, used for creating `qns_key`).

**Logic:**
The `DataLoader` finds the index of the correct answer by comparing the `answer` column string with the content of `a0`...`a4`.

---

## 3. Video Features (ViT/Global)
**Path:** `video_feature_path/{split}/{video_id}.pt`

*   **Format:** PyTorch Tensor (`.pt`) or Numpy Array inside the file.
*   **Content:** Global frame-level features extracted from a model like ViT or ResNet.
*   **Shape:** `[Frames, Dim]` 
    *   Example: `[16, 768]` or `[32, 4096]`.
    *   The model expects `frame_feat_dim` to match the 2nd dimension here.

---

## 4. Object Features (R-CNN/Detector)
**Path:** `object_feature_path/{video_id}/{frame_filename}.pkl`

*   **Structure:** One folder per video, containing multiple `.pkl` files (one per sampled frame).
*   **Ordering:** Files are sorted by the number in their filename (e.g., `image_01.pkl` comes before `image_10.pkl`).

**File Content (.pkl):**
Each `.pkl` file should contain a dictionary (or tuple) with at least two keys/elements:
1.  **Features (`feat`):** Appearance features of the detected objects.
    *   **Shape:** `[Num_Objects, Feature_Dim]` (e.g., `[20, 2048]`).
    *   These are usually extracted from the ROI-Pooling/ROI-Align layer of a detector (Faster R-CNN).
2.  **Bounding Boxes (`bbox`):** Spatial coordinates of the objects.
    *   **Shape:** `[Num_Objects, 4]`.
    *   **Format:** Absolute pixel coordinates `[x1, y1, x2, y2]`.

**Process in DataLoader:**
1.  Loads `feat` and `bbox`.
2.  Normalizes `bbox` by image `width`/`height` and adds relative area -> creates 5D vector.
3.  Concatenates `feat` (2048) + `bbox` (5) -> `[Num_Objects, 2053]`.
4.  Stacks frames -> `[Frames, Num_Objects, 2053]`.

---

## 5. Splits (Optional)
**Path:** `split_dir/{split}.txt`

*   **Content:** Simple text file with one `video_id` per line.
*   **Purpose:** If provided, the DataLoader filters the CSV to only include rows where `video_id` exists in this text file.
