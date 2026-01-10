# CausalVidQA - Video Question Answering with TranSTR

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/yl3800/IGV) 

</div>

---

## 📊 Dataset Statistics

### CausalVidQA Dataset

| Split | Videos | Samples (qtype=-1) | Mô tả |
|-------|--------|-------------------|-------|
| Train | ~8,000 | ~48,000 | Training set |
| Valid | 2,695 | 16,170 | Validation set |
| Test | 5,429 | 32,574 | Test set |

> **⚠️ Lưu ý**: Dataset `dataset-split-1` trên Kaggle có vấn đề - file `train.pkl` chỉ chứa **1 video**! Cần sử dụng dataset split đầy đủ hoặc tự tạo lại file split.

### Question Types (6 loại - mỗi video có 6 câu hỏi)

| qtype | Tên | Mô tả | Ví dụ |
|-------|-----|-------|-------|
| 0 | **Descriptive** | Mô tả hành động/sự kiện | "What is the person doing?" |
| 1 | **Explanatory** | Giải thích nguyên nhân | "Why did the person do that?" |
| 2 | **Predictive Answer** | Dự đoán kết quả | "What will happen next?" |
| 3 | **Predictive Reason** | Lý do dự đoán | "Why will that happen?" |
| 4 | **Counterfactual Answer** | Kết quả giả định | "What would happen if...?" |
| 5 | **Counterfactual Reason** | Lý do giả định | "Why would that happen?" |

### Visual Features (what the code actually uses)

| Tensor | Shape | Produced from |
|--------|-------|---------------|
| Appearance | `(T, 2048)` or `(T, N, 2048)` | `appearance_feat.h5` (ResNet-101), mean over N if present |
| Motion | `(T, 2048)` or `(T, N, 2048)` | `motion_feat.h5` (3D ResNet), mean over N if present |
| Frame (combined) | `(T, 4096)` | concat(appearance, motion) |
| “Object” | `(T, O, 2053)` | appearance tiled O times + dummy full-frame bbox (no detector) |

---

## 📁 Cấu trúc dữ liệu

```
kaggle-input/
├── visual-feature/
│   ├── appearance_feat.h5    # (N_videos, T, 2048) hoặc (N_videos, T, C, 2048)
│   ├── motion_feat.h5        # (N_videos, T, 2048)
│   └── idx2vid.pkl           # List[video_id] - index to video_id mapping
│
├── text-annotation/
│   └── QA/
│       ├── video_id_1/
│       │   ├── text.json     # Questions và candidate answers
│       │   └── answer.json   # Ground truth answers (0-4)
│       └── video_id_2/
│           └── ...
│
└── dataset-split-1/
    ├── train.pkl             # List[video_id] cho training
    ├── valid.pkl             # List[video_id] cho validation  
    └── test.pkl              # List[video_id] cho testing
```

---

## 🔄 End-to-end flow (what the current code does)

1) **Inputs on disk**
   - `appearance_feat.h5`, `motion_feat.h5`, `idx2vid.pkl`
   - `text-annotation/QA/<video_id>/{text.json, answer.json}`
   - Split files `{train,valid/test,test}.pkl`

2) **Dataset construction (`VideoQADataset` in `DataLoader.py`)**
   - Load video IDs from split pkl (optionally truncated by `max_samples`).
   - Map video IDs → feature indices via `idx2vid.pkl`.
   - Load appearance and motion features for the videos; if a 3rd dim exists, mean-pool over it.
   - Build sample list by `qtype`:
     - `-1`: all qtypes 0–5
     - `0` or `1`: single type
     - `2`: predictive answer + predictive reason (2,3)
     - `3`: counterfactual answer + counterfactual reason (4,5)  ⚠️ naming quirk
   - For each sample: load question/answers from `text.json` + labels from `answer.json`.

3) **Feature preparation (per sample)**
   - Frame feature: concat appearance + motion → `(T, 4096)`.
   - “Object” feature: tile appearance to `(T, O, 2048)`, append dummy bbox `[0,0,1,1,1]` → `(T, O, 2053)`.  
     *No object detector is run; boxes are full-frame placeholders.*
   - Answers are formatted as `[CLS] question [SEP] candidate_i`.
   - Outputs per item: `(vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key)`.

4) **Dataloaders**
   - `train/val/test` loaders with given batch size, optional workers; `pin_memory=True`.

5) **Model (`networks/model.py`)**
   - Encodes video features and text; uses transformer decoder for answer scoring.
   - Expects both frame and “object” tensors even though objects are synthesized as above.

6) **Training loop (`transtr.ipynb`)**
   - Supports gradient accumulation, LR scheduling, early stopping, W&B logging.
   - Tracks per-qtype accuracy and wrong samples.

7) **Evaluation / prediction**
   - Best checkpoint evaluated on test loader.
   - Predictions saved to JSON; per-qtype metrics computed; wrong samples logged/saved.

---

### Text Annotation Format

**text.json:**
```json
{
  "descriptive": {
    "question": "What is the man doing?",
    "answer": ["Walking", "Running", "Sitting", "Standing", "Jumping"]
  },
  "explanatory": {
    "question": "Why is the man running?",
    "answer": ["To catch bus", "To exercise", "Being chased", "Late for work", "For fun"]
  },
  "predictive": {
    "question": "What will happen next?",
    "answer": ["He will stop", "He will fall", "He will continue", "He will turn", "He will sit"],
    "reason": ["He is tired", "Road is slippery", "He has energy", "He sees something", "Reached destination"]
  },
  "counterfactual": {
    "question": "What would happen if he stopped?",
    "answer": ["Miss the bus", "Rest", "Fall down", "Get caught", "Nothing"],
    "reason": ["Bus leaves", "He is tired", "Momentum", "Chaser catches up", "No effect"]
  }
}
```

**answer.json:**
```json
{
  "descriptive": {"answer": 1},
  "explanatory": {"answer": 0},
  "predictive": {"answer": 2, "reason": 2},
  "counterfactual": {"answer": 0, "reason": 0}
}
```

---

## 🔧 Cài đặt

```bash
cd causalvid
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.11+
- transformers
- h5py
- einops
- numpy

---

## 📥 Download dữ liệu

### Option 1: Kaggle API
```python
import kagglehub

visual_feature_path = kagglehub.dataset_download('lusnaw/visual-feature')
split_path = kagglehub.dataset_download('lusnaw/dataset-split-1')
text_annotation_path = kagglehub.dataset_download('lusnaw/text-annotation')
```

### Option 2: Kaggle Notebook
Thêm datasets vào notebook:
- `lusnaw/visual-feature`
- `lusnaw/text-annotation`
- `lusnaw/dataset-split-1`

---

## 🚀 Training

### Command Line

```bash
python train.py \
    -v causalvid_full \
    -bs 16 \
    -lr 1e-4 \
    -epoch 20 \
    -gpu 0 \
    --sample_list_path "/path/to/dataset-split-1" \
    --video_feature_path "/path/to/visual-feature" \
    --text_annotation_path "/path/to/text-annotation" \
    --qtype -1 \
    -fk 8 -ok 5 -objs 20 \
    -el 1 -dl 1 \
    -t microsoft/deberta-base
```

### Kaggle Notebook
Sử dụng `train_causalvid.ipynb`:
1. Install dependencies
2. Patch DataLoader (xử lý dimension mismatch)
3. Configure hyperparameters
4. Train model

### Tham số chính

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `-v` | Tên experiment | required |
| `-bs` | Batch size | 16 |
| `-lr` | Learning rate | 1e-4 |
| `-epoch` | Số epochs | 20 |
| `--qtype` | Loại câu hỏi (-1=all, 0-5=specific) | -1 |
| `--max_samples` | Giới hạn số video (None=all) | None |
| `-fk` | Top-K frames | 8 |
| `-ok` | Top-K objects | 5 |
| `-objs` | Số objects/frame | 20 |
| `-t` | Text encoder | microsoft/deberta-base |

---

## 🧪 Evaluation

### Chạy đánh giá

```bash
python test.py \
    -v eval_test \
    -bs 32 \
    --sample_list_path "/path/to/dataset-split-1" \
    --video_feature_path "/path/to/visual-feature" \
    --text_annotation_path "/path/to/text-annotation" \
    --qtype -1 \
    --model_path "./models/best_model-xxx.ckpt"
```

### Evaluation Metrics

| Metric | Mô tả | Cách tính |
|--------|-------|-----------|
| **Des** | Descriptive accuracy | Đúng/Tổng samples với qtype=0 |
| **Exp** | Explanatory accuracy | Đúng/Tổng samples với qtype=1 |
| **Pred-A** | Predictive Answer | Đúng/Tổng samples với qtype=2 |
| **Pred-R** | Predictive Reason | Đúng/Tổng samples với qtype=3 |
| **CF-A** | Counterfactual Answer | Đúng/Tổng samples với qtype=4 |
| **CF-R** | Counterfactual Reason | Đúng/Tổng samples với qtype=5 |
| **Pred** | Predictive Combined | Cả answer VÀ reason đúng cho cùng video |
| **CF** | Counterfactual Combined | Cả answer VÀ reason đúng cho cùng video |
| **ALL** | Overall | (Des + Exp + Pred + CF) / 4 |

### Evaluation Script

```python
from eval_mc import accuracy_metric_cvid

# Đánh giá từ file prediction
accuracy_metric_cvid('./prediction/result.json')

# Output example:
# Des: 45.32%
# Exp: 38.21%
# Pred-A: 42.15%  Pred-R: 35.67%  Pred: 28.43%
# CF-A: 40.89%    CF-R: 33.45%    CF: 25.12%
# ALL: 34.22%
```

### Giải thích cách tính Pred và CF

```
Pred (Combined) = Số videos có CẢ Pred-A VÀ Pred-R đúng / Tổng videos
CF (Combined) = Số videos có CẢ CF-A VÀ CF-R đúng / Tổng videos

ALL = (Des + Exp + Pred + CF) / 4
    = (45.32 + 38.21 + 28.43 + 25.12) / 4
    = 34.27%
```

---

## 📊 Expected Results

### Baseline (Random)
- 5-way multiple choice: ~20% accuracy

### Trained Model
| Metric | Expected Range |
|--------|----------------|
| Des | 45-55% |
| Exp | 35-45% |
| Pred | 25-35% |
| CF | 20-30% |
| ALL | 30-40% |

---

## ⚠️ Known Issues & Solutions

### 1. Train split chỉ có 1 video
**Vấn đề**: `train.pkl` trên Kaggle `dataset-split-1` chỉ chứa 1 video

**Giải pháp**:
```python
# Option 1: Swap train với valid để test
train_split = 'valid'  
val_split = 'test'     

# Option 2: Tự tạo train.pkl từ toàn bộ videos
import pickle
# Load idx2vid.pkl để lấy tất cả video IDs
# Chia theo tỷ lệ 70/15/15 cho train/val/test
```

### 2. DeBERTa FP16 Overflow
**Vấn đề**: `RuntimeError: value cannot be converted to type at::Half`

**Giải pháp**: Tắt mixed precision
```python
USE_AMP = False
```

### 3. Multiprocessing Error trên Kaggle
**Vấn đề**: `Bad file descriptor` với num_workers > 0

**Giải pháp**: 
```python
DataLoader(..., num_workers=0)
```

### 4. Dimension Mismatch giữa app và mot features
**Vấn đề**: `appearance_feat` có 3 dims, `motion_feat` có 2 dims

**Giải pháp** (đã patch trong DataLoader):
```python
if app_feat.ndim == 3:
    app_feat = app_feat.mean(axis=1)
if mot_feat.ndim == 3:
    mot_feat = mot_feat.mean(axis=1)
```

---

## 📂 Output Files

```
causalvid/
├── models/
│   └── best_model-{version}.ckpt    # Model checkpoint
├── prediction/
│   └── {version}-{epoch}-{acc}.json # Predictions
└── log/
    └── {version}.log                # Training log
```

### Prediction JSON Format
```json
{
  "video_id_0": {"prediction": 2, "answer": 2, "qtype": 0},
  "video_id_1": {"prediction": 0, "answer": 1, "qtype": 1},
  ...
}
```

---

## 🏗️ Model Architecture - Answer Decoder

### Tổng quan Answer Decoder

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ANSWER DECODER ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    INPUTS
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  Video Memory │           │ Answer Query  │           │  Query Mask   │
│   (v_mem)     │           │   (a_query)   │           │   (q_mask)    │
│ [B, T, d_model]│          │[B*5, L, d_model]│         │  [B*5, L]     │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        │                           ▼                           │
        │                   ┌───────────────┐                   │
        │                   │  [CLS] Token  │                   │
        │                   │   Extraction  │                   │
        │                   │  a_query[:,0] │                   │
        │                   └───────┬───────┘                   │
        │                           │                           │
        │                           ▼                           │
        │               ┌───────────────────────┐               │
        │               │    Expand v_mem       │               │
        │               │  repeat for 5 answers │               │
        │               │   [B, T, D] →         │               │
        │               │   [B*5, T, D]         │               │
        │               └───────────┬───────────┘               │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORMER DECODER LAYER                               │
│                              (num_layers = 1)                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │   ┌─────────────────┐                                                     │  │
│  │   │  Self-Attention │  ← Answer query attends to itself                   │  │
│  │   │   (masked)      │                                                     │  │
│  │   └────────┬────────┘                                                     │  │
│  │            │                                                              │  │
│  │            ▼                                                              │  │
│  │   ┌─────────────────┐                                                     │  │
│  │   │ Cross-Attention │  ← Answer query attends to video memory             │  │
│  │   │  Q: a_query     │                                                     │  │
│  │   │  K: v_mem       │                                                     │  │
│  │   │  V: v_mem       │                                                     │  │
│  │   └────────┬────────┘                                                     │  │
│  │            │                                                              │  │
│  │            ▼                                                              │  │
│  │   ┌─────────────────┐                                                     │  │
│  │   │   Feed Forward  │                                                     │  │
│  │   │     Network     │                                                     │  │
│  │   └────────┬────────┘                                                     │  │
│  │            │                                                              │  │
│  └────────────┼──────────────────────────────────────────────────────────────┘  │
│               │                                                                 │
└───────────────┼─────────────────────────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Decoder Output│
        │ [B*5, L, D]   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Take [CLS]   │
        │   Position    │
        │  output[:,0]  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Reshape     │
        │ [B*5, D] →    │
        │ [B, 5, D]     │
        └───────┬───────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ANSWER CLASSIFIER                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │   ┌─────────────────┐                           ┌─────────────────┐       │  │
│  │   │  Linear Layer   │  ────────────────────────►│  Linear Layer   │       │  │
│  │   │   (D → 1)       │                           │  (squeeze)      │       │  │
│  │   └─────────────────┘                           └─────────────────┘       │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                              [B, 5, D] → [B, 5, 1] → [B, 5]                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │    OUTPUT     │
        │  Logits [B,5] │
        │  (5 answers)  │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Softmax     │
        │  (in loss)    │
        │ → Prediction  │
        └───────────────┘
```

### Parameter Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 768 | Hidden dimension |
| `nheads` | 8 | Attention heads |
| `num_decoder_layers` | 1 | Number of decoder layers |
| `dropout` | 0.1 | Dropout rate |
| `activation` | relu | Activation function |
| `n_query` | 5 | Number of answer choices |

### Data Flow Example (Batch size B=16)

```
1. INPUT:
   ├── v_mem: [16, T, 768]      # Video memory from encoder
   ├── a_query: [80, L, 768]    # 16*5 answer embeddings  
   └── q_mask: [80, L]          # Answer attention masks

2. EXPAND VIDEO MEMORY:
   v_mem: [16, T, 768] → repeat → [80, T, 768]
   (Each video paired with 5 answers)

3. TRANSFORMER DECODER:
   ├── Self-Attention: a_query attends to a_query
   ├── Cross-Attention: a_query attends to v_mem
   └── FFN: position-wise feed-forward
   Output: [80, L, 768]

4. EXTRACT [CLS]:
   output[:,0,:]: [80, 768]

5. RESHAPE:
   [80, 768] → [16, 5, 768]

6. CLASSIFIER:
   [16, 5, 768] → Linear → [16, 5, 1] → squeeze → [16, 5]

7. OUTPUT:
   Logits: [16, 5] (score for each of 5 answers)
   Prediction: argmax → answer index (0-4)
```

### Cross-Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CROSS-ATTENTION DETAIL                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│         Answer Query                          Video Memory                      │
│         [B*5, L, D]                           [B*5, T, D]                       │
│              │                                     │                            │
│              ▼                                     ▼                            │
│         ┌────────┐                           ┌──────────┐                       │
│         │   Wq   │                           │  Wk, Wv  │                       │
│         └────┬───┘                           └────┬─────┘                       │
│              │                                    │                             │
│              ▼                                    ▼                             │
│         Q [B*5, L, D]                    K, V [B*5, T, D]                       │
│              │                                    │                             │
│              └──────────────┬─────────────────────┘                             │
│                             │                                                   │
│                             ▼                                                   │
│                   ┌─────────────────────┐                                       │
│                   │  Attention Scores   │                                       │
│                   │  Q @ K^T / sqrt(d)  │                                       │
│                   │  [B*5, L, T]        │                                       │
│                   └──────────┬──────────┘                                       │
│                              │                                                  │
│                              ▼                                                  │
│                   ┌─────────────────────┐                                       │
│                   │      Softmax        │                                       │
│                   │  [B*5, L, T]        │                                       │
│                   └──────────┬──────────┘                                       │
│                              │                                                  │
│                              ▼                                                  │
│                   ┌─────────────────────┐                                       │
│                   │  Attention @ V      │                                       │
│                   │  [B*5, L, D]        │                                       │
│                   └──────────┬──────────┘                                       │
│                              │                                                  │
│                              ▼                                                  │
│                   Answer-aware Video                                            │
│                      Representation                                             │
│                                                                                 │
│  ═══════════════════════════════════════════════════════════════════════════   │
│  Ý nghĩa: Mỗi answer "nhìn" vào video để tìm evidence hỗ trợ                    │
│  → Answer đúng sẽ có attention cao vào frames liên quan                         │
│  → Answer sai sẽ có attention thấp hoặc không phù hợp                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 References

- [IGV Paper (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
- [CausalVidQA Dataset](https://github.com/bcmi/Causal-VidQA)
- [Original IGV Code](https://github.com/yl3800/IGV)

---

## 📝 Citation

```bibtex
@InProceedings{Li_2022_CVPR,
    author    = {Li, Yicong and Wang, Xiang and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title     = {Invariant Grounding for Video Question Answering},
    booktitle = {CVPR},
    year      = {2022},
    pages     = {2928-2937}
}
```