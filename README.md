# TranSTR-DN: Spatio-Temporal Rationalization for Causal Video Question Answering with DINOv3 Visual Features and Noise-Aware Learning

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/DanielQH07/tranSTR_Casual) 

</div>

---

## 1. Introduction

Video Question Answering (VideoQA), đặc biệt trên bộ dữ liệu **CausalVidQA** — yêu cầu mô hình không chỉ mô tả mà còn phải **suy luận nhân quả** (explanatory, predictive, counterfactual) — đặt ra hai thách thức lớn:

1. **Visual representation chưa đủ mạnh**: Baseline TranSTR sử dụng đặc trưng concat `(ResNet-101 appearance + 3D-ResNet motion)` với dimension `4096`. Biểu diễn này (i) không mang semantic cấp cao cần thiết cho suy luận nhân quả, (ii) trộn lẫn hai nguồn feature bằng phép nối thô, và (iii) thiếu khả năng nắm bắt fine-grained visual detail.

2. **Label noise trong dataset**: CausalVidQA chứa các câu hỏi suy luận chủ quan (explanatory, counterfactual) mà ground-truth thường mơ hồ hoặc có nhiều đáp án hợp lý. Việc ép mô hình học cứng trên nhãn nhiễu dẫn đến overfit noise thay vì học được reasoning pattern.

Chúng tôi đề xuất **TranSTR-DN** (TranSTR with DINOv3 and NCOD), giải quyết đồng thời cả hai vấn đề trên bằng:
- **Thay thế visual backbone** bằng DINOv3 — vision foundation model train bằng self-supervised learning, cung cấp biểu diễn ngữ nghĩa mạnh hơn.
- **Tích hợp NCOD** (Noisy Correspondence Detection) — cơ chế bi-level optimization học tham số chiết khấu (discount parameter) U cho mỗi sample, tự động phát hiện và giảm ảnh hưởng của mẫu nhiễu.

---

## 2. Proposed Method

### 2.1 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           TranSTR-DN Architecture Overview                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────────┐      ┌──────────────────┐
                     │   Raw Video      │      │  Question + 5    │
                     │   Frames         │      │  Answer Choices  │
                     └────────┬─────────┘      └────────┬─────────┘
                              │                         │
                 ┌────────────┼───────────┐             │
                 │            │           │             │
                 ▼            ▼           ▼             ▼
         ┌──────────┐  ┌──────────┐ ┌─────────┐ ┌─────────────┐
         │ DINOv3   │  │  Object  │ │ Object  │ │  DeBERTa    │
         │ ViT-L/14 │  │ Detector │ │ Feature │ │  Encoder    │
         │ (NEW)    │  │ (DETR)   │ │ + BBox  │ │             │
         └────┬─────┘  └────┬─────┘ └───┬─────┘ └──────┬──────┘
              │              │           │              │
              ▼              └─────┬─────┘              ▼
     ┌────────────────┐            │           ┌────────────────┐
     │ Frame Features │    ┌───────▼───────┐   │ Text Features  │
     │ [T, 1024]      │    │ Obj Features  │   │ q: [L_q, 768]  │
     │                │    │ [T, O, 2053]  │   │ a: [5, L_a,768]│
     └───────┬────────┘    └───────┬───────┘   └───────┬────────┘
             │                     │                    │
             ▼                     ▼                    ▼
     ┌────────────────┐    ┌──────────────┐    ┌───────────────┐
     │  FeatureResizer│    │FeatureResizer│    │  text_proj    │
     │  1024 → 768    │    │ 2053 → 768   │    │  768 → 768    │
     └───────┬────────┘    └──────┬───────┘    └───────┬───────┘
             │                    │                     │
             ║                    ║                     ║
    ═════════╬════════════════════╬═════════════════════╬════════════════
             ║    HIERARCHICAL SPATIO-TEMPORAL RATIONALIZATION          
    ═════════╬════════════════════╬═════════════════════╬════════════════
             ▼                    │                     │
     ┌───────────────┐            │                     │
     │ Frame Decoder │◄───────────┼─────────────────────┤
     │ (cross-attn   │            │          q_local as memory
     │  with q_local)│            │                     │
     └───────┬───────┘            │                     │
             │ frame_att          │                     │
             ▼                    │                     │
     ┌───────────────┐            │                     │
     │ PerturbedTopK │            │                     │
     │ (select K=5   │            │                     │
     │  frames)      │            │                     │
     └───────┬───────┘            │                     │
             │ idx_frame          │                     │
             ├────────────────────┤                     │
             ▼                    ▼                     │
     ┌──────────────┐    ┌──────────────┐               │
     │ Selected     │    │ Selected Obj │               │
     │ Frame Feats  │    │ Features     │               │
     │ [B,K_f,d]    │    │ [B,K_f,O,d]  │              │
     └──────┬───────┘    └──────┬───────┘               │
             │                   ▼                      │
             │           ┌───────────────┐              │
             │           │  Obj Decoder  │◄─────────────┤
             │           │ (cross-attn   │     q_local as memory
             │           │  with q_local)│              │
             │           └───────┬───────┘              │
             │                   │ obj_att               │
             │                   ▼                      │
             │           ┌───────────────┐              │
             │           │ PerturbedTopK │              │
             │           │ (select K=12  │              │
             │           │  objects)     │              │
             │           └───────┬───────┘              │
             │                   │                      │
             ▼                   ▼                      │
     ┌─────────────────────────────────┐                │
     │       Frame-Object Decoder      │                │
     │  (frame queries attend to       │                │
     │   selected object features)     │                │
     └──────────────┬──────────────────┘                │
                    │                                   │
                    ▼                                   │
     ┌─────────────────────────────────┐                │
     │     Vision-Language Encoder     │◄───────────────┘
     │  concat(frame_obj, q_local)     │       q_local
     │  + positional encoding          │
     └──────────────┬──────────────────┘
                    │ v_mem [B, K_f+L_q, d]
                    │
    ════════════════╬═══════════════════════════════════════════
                    ║          ANSWER DECODING + NCOD
    ════════════════╬═══════════════════════════════════════════
                    ▼
     ┌─────────────────────────────────┐
     │        Answer Decoder           │
     │  tgt: [CLS] of each QA pair    │
     │  memory: v_mem                  │
     │  → cross-attention scoring      │
     └──────────────┬──────────────────┘
                    │
                    ▼
     ┌─────────────────────────────────┐
     │       Classifier (Linear)       │
     │  [B, 5, d] → [B, 5, 1] → [B,5]│
     └──────────────┬──────────────────┘
                    │ logits
                    │
         ┌──────────┼──────────┐
         │                     │
         ▼                     ▼
  ┌──────────────┐     ┌──────────────┐
  │  L₁ Loss     │     │  L₂ Loss     │
  │ (update θ)   │     │ (update U)   │
  │              │     │              │
  │ CE with      │     │ MSE with     │
  │ U.detach()   │     │ probs.detach │
  │ shift        │     │ + U shift    │
  └──────┬───────┘     └──────┬───────┘
         │                     │
    opt_model.step()      opt_U.step()
                          U.clamp_(0, 0.99)
```

### 2.2 Thay đổi Visual Backbone: DINOv3

#### 2.2.1 Motivation

| Tiêu chí | ResNet-101 + 3D-ResNet (Baseline) | DINOv3 ViT-L/14 (Ours) |
|-----------|-----------------------------------|------------------------|
| **Paradigm** | Supervised (ImageNet / Kinetics) | Self-supervised (LVD-142M) |
| **Feature dim** | 4096 (concat 2048+2048) | 1024 |
| **Semantic level** | Low-level texture + motion | High-level semantic |
| **Object awareness** | Không phân biệt object | Attention map tự nhiên trên objects |
| **Efficiency** | 2× parameters cho 2 backbone | 1 backbone duy nhất |

**Lý do chọn DINOv3 thay vì CLIP hay ViT supervised:**

1. **Self-supervised features mang tính tổng quát cao hơn**: DINOv3 được train trên 142M ảnh bằng self-supervised distillation, không bị bias vào label categories cụ thể. Điều này quan trọng cho CausalVidQA vì các câu hỏi counterfactual/predictive yêu cầu suy luận trên visual concepts chưa từng được label.

2. **Attention map phong phú**: DINOv3 ViT tạo ra attention map tự nhiên trên semantic regions, giúp các module TopK frame selection và object selection của TranSTR hoạt động hiệu quả hơn — thay vì chỉ dựa vào feature magnitude.

3. **Giảm feature dimension**: Từ `4096` xuống `1024`, giảm computational cost tại `FeatureResizer` (projection layer) mà vẫn giữ hoặc tăng chất lượng biểu diễn. Điều này cho phép tăng batch size hoặc tăng số encoder/decoder layers trong cùng VRAM budget.

#### 2.2.2 Feature Extraction Pipeline

```
Video → Sample 16 frames → DINOv3 ViT-L/14 → [CLS] token per frame
                                                      ↓
                                              [16, 1024] per video
                                                      ↓
                                              Save as {video_id}.pt
```

Mỗi video được biểu diễn bằng tensor `[T, 1024]` (T=16 frames), với mỗi frame feature là `[CLS]` token output từ DINOv3 ViT-L/14. Feature extraction chạy offline, kết quả lưu dạng `.pt` file được chia theo split (`train/`, `valid/`, `test/`) rồi merge vào thư mục chung.

#### 2.2.3 Integration vào Model

Thay đổi duy nhất trong kiến trúc model:

```python
# Baseline TranSTR
self.frame_resize = FeatureResizer(
    input_feat_size=4096,   # concat(appearance, motion)
    output_feat_size=768,
    dropout=0.1
)

# TranSTR-DN (Ours)
self.frame_resize = FeatureResizer(
    input_feat_size=1024,   # DINOv3 [CLS] token
    output_feat_size=768,
    dropout=0.3
)
```

Toàn bộ kiến trúc downstream (frame decoder, object decoder, frame-object decoder, VL encoder, answer decoder) **giữ nguyên không thay đổi**.

### 2.3 Tích hợp NCOD: Noisy Correspondence Detection

#### 2.3.1 Motivation

CausalVidQA chứa 6 loại câu hỏi, trong đó 4 loại suy luận (explanatory, predictive answer/reason, counterfactual answer/reason) mang tính chủ quan cao. Ví dụ:

> **Q**: "Why did the person pick up the phone?" 
> **Choices**: (A) To make a call (B) Phone was ringing (C) Habit (D) To check time (E) Curious
> **Ground truth**: B

Trong trường hợp trên, đáp án A, B, và E đều có thể hợp lý tùy ngữ cảnh. Khi mô hình bị ép học rằng **chỉ B là đúng**, nó sẽ overfit vào pattern nhiễu thay vì học reasoning thực.

#### 2.3.2 NCOD Formulation

Chúng tôi tích hợp cơ chế **tham số chiết khấu U** (discount parameter) từ bài báo NCOD, biến bài toán training thành **bi-level optimization**:

**Level 1 — Cập nhật model parameters θ (L₁ loss):**

Với mỗi sample `i`, xác suất shifted được tính:

```
p̃ᵢ = softmax(logits_i) + U[i].detach() × y_onehot_i
```

Trong đó `U[i]` là tham số chiết khấu riêng cho sample `i`, được **detach** (không tính gradient qua U khi cập nhật θ). Loss L₁ là negative log-likelihood trên xác suất shifted:

```
L₁ = -𝔼ᵢ [ Σⱼ y_ij × log(clamp(p̃ᵢⱼ, ε, 1)) ]
```

**Ý nghĩa**: Khi U[i] lớn (sample nhiễu), `p̃ᵢ` được "nâng lên" gần 1.0 tại vị trí ground truth → loss nhỏ → model **bỏ qua** sample này. Khi U[i] nhỏ (sample sạch), model học bình thường.

**Level 2 — Cập nhật U (L₂ loss):**

```
p̃ᵢ = softmax(logits_i).detach() + U[i] × y_onehot_i
L₂ = MSE(p̃ᵢ, y_onehot_i)
```

Ở đây `softmax(logits)` được **detach** (không tính gradient qua θ khi cập nhật U). Ý nghĩa: nếu model đã dự đoán đúng (softmax ≈ y_onehot), U không cần lớn → U giảm. Nếu model liên tục sai trên sample này, U phải tăng để "bù" khoảng cách giữa prediction và label.

**Theorem 5.1 ràng buộc**: U[i] ∈ [0, 0.99], được enforce bằng `U.clamp_(0, 0.99)` sau mỗi optimizer step.

#### 2.3.3 Bi-level Optimization với Gradient Accumulation

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Training Loop (per epoch)                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  for each micro-batch (bs=8):                                         │
│    1. Forward:  logits = model(ff, of, q, a)                          │
│    2. Compute L₁/accumulation_steps → backward (accumulate θ grads)   │
│    3. Compute L₂/accumulation_steps → backward (accumulate U grads)   │
│                                                                        │
│    if (batch_idx + 1) % accumulation_steps == 0:   # every 4 batches  │
│      4. clip_grad_norm_(model.parameters(), max_norm=1.0)             │
│      5. opt_model.step() → update θ                                   │
│      6. opt_U.step() → update U                                       │
│      7. U.clamp_(0, 0.99)                                             │
│      8. zero_grad() for both optimizers                               │
│                                                                        │
│  Effective batch size = 8 × 4 = 32                                    │
│  VRAM usage ≈ same as bs=8 (chỉ tích lũy gradient, không tích lũy    │
│  activations)                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

#### 2.3.4 Dual Optimizer Design

| Optimizer | Target | Algorithm | Learning Rate | Ghi chú |
|-----------|--------|-----------|---------------|---------|
| `opt_model` | θ (model params) | AdamW | 1e-5 (text encoder: 1e-5) | weight_decay=1e-4 |
| `opt_U` | U[N_train] | SGD | 0.1 | Không momentum, theo paper appendix |

Tách hai optimizer đảm bảo:
- Model parameters được optimize với adaptive learning rate (AdamW) phù hợp cho deep networks.
- U được optimize với SGD đơn giản, learning rate cao (0.1) để nhanh chóng phân biệt clean vs. noisy samples.

### 2.4 Object Feature Pipeline

Song song với DINOv3 frame features, mô hình vẫn sử dụng **object features** từ object detector (DETR) với format:

| Tensor | Shape | Nguồn |
|--------|-------|-------|
| Object features | `(T, O, 2048)` | DETR ResNet-50 backbone |
| Bounding boxes | `(T, O, 5)` | DETR bboxes, transformed via `transform_bb` |
| Combined | `(T, O, 2053)` | concat(features, bbox) |

Object features cung cấp thông tin **spatial fine-grained** mà DINOv3 frame-level [CLS] token không capture được. Sự kết hợp DINOv3 (semantic frame) + DETR (spatial object) tạo ra biểu diễn **complementary**:
- DINOv3 frame features → temporal rationalization (chọn frames quan trọng)
- DETR object features → spatial rationalization (chọn objects quan trọng trong frames đã chọn)

---

## 3. Experimental Setup

### 3.1 Dataset: CausalVidQA

| Split | Videos | Samples (qtype=-1) | Mô tả |
|-------|--------|---------------------|-------|
| Train | ~8,000 | ~48,000 | Training set |
| Valid | 2,695 | 16,170 | Validation set |
| Test | 5,429 | 32,574 | Test set |

### Question Types (6 loại — mỗi video có 6 câu hỏi)

| qtype | Tên | Mô tả | Ví dụ |
|-------|-----|-------|-------|
| 0 | **Descriptive** | Mô tả hành động/sự kiện | "What is the person doing?" |
| 1 | **Explanatory** | Giải thích nguyên nhân | "Why did the person do that?" |
| 2 | **Predictive Answer** | Dự đoán kết quả | "What will happen next?" |
| 3 | **Predictive Reason** | Lý do dự đoán | "Why will that happen?" |
| 4 | **Counterfactual Answer** | Kết quả giả định | "What would happen if...?" |
| 5 | **Counterfactual Reason** | Lý do giả định | "Why would that happen?" |

### 3.2 Implementation Details

| Hyperparameter | Value | Ghi chú |
|----------------|-------|---------|
| Frame feature backbone | DINOv3 ViT-L/14 | `frame_feat_dim = 1024` |
| Object feature | DETR ResNet-50 | `obj_feat_dim = 2053` |
| Text encoder | DeBERTa-base | `microsoft/deberta-base` |
| d_model | 768 | Hidden dimension |
| Attention heads | 8 | Multi-head attention |
| Encoder layers | 2 | VL encoder |
| Decoder layers | 2 | Frame/Obj/FO/Answer decoders |
| Dropout | 0.3 | Increased vs baseline 0.1 |
| Activation | GELU | Smoother than ReLU |
| Micro batch size | 8 | Per-GPU |
| Gradient accumulation | 4 steps | Effective BS = 32 |
| Learning rate (model) | 1e-5 | AdamW, weight_decay=1e-4 |
| Learning rate (text encoder) | 1e-5 | Same param group |
| NCOD U learning rate | 0.1 | SGD, per paper appendix |
| NCOD U init | mean=1e-8, std=1e-9 | Near-zero initialization |
| NCOD U clamp | [0, 0.99] | Theorem 5.1 bound |
| TopK frames | 5 (select from 16) | PerturbedTopK |
| TopK objects | 12 (select from 20) | PerturbedTopK |
| Epochs | 20 | With ReduceLROnPlateau |
| Patience | 5 | LR scheduler |
| Gradient clipping | max_norm=1.0 | Model params only |

### 3.3 Evaluation Metrics

| Metric | Mô tả | Cách tính |
|--------|-------|-----------|
| **Des** | Descriptive accuracy | Đúng/Tổng samples qtype=0 |
| **Exp** | Explanatory accuracy | Đúng/Tổng samples qtype=1 |
| **Pred-A** | Predictive Answer | Đúng/Tổng samples qtype=2 |
| **Pred-R** | Predictive Reason | Đúng/Tổng samples qtype=3 |
| **CF-A** | Counterfactual Answer | Đúng/Tổng samples qtype=4 |
| **CF-R** | Counterfactual Reason | Đúng/Tổng samples qtype=5 |
| **PAR** | Predictive (hard) | Cả answer VÀ reason đúng cho cùng video |
| **CAR** | Counterfactual (hard) | Cả answer VÀ reason đúng cho cùng video |
| **Acc (ALL)** | Overall | (Des + Exp + PAR + CAR) / 4 |

---

## 4. Data Structure

### 4.1 Directory Layout

```
kaggle-input/
├── dinov3-feat/                          # 🆕 DINOv3 features (thay thế appearance+motion)
│   └── features/
│       ├── train/                        #   Per-split extraction
│       │   ├── video_id_1.pt             #   [T, 1024] tensor
│       │   ├── video_id_2.pt
│       │   └── ...
│       ├── valid/
│       └── test/
│
├── object-detection-causal-full/         # Object features (DETR)
│   ├── features_node_0/                  #   Kaggle subdirectory structure
│   │   ├── video_id_1.pkl               #   {'features': [T,O,2048], 'bboxes': [T,O,4]}
│   │   └── ...
│   └── features_node_1/
│
├── text-annotation/
│   └── QA/
│       ├── video_id_1/
│       │   ├── text.json                 # Questions + candidate answers
│       │   └── answer.json               # Ground truth (0-4)
│       └── ...
│
└── casual-vid-data-split/
    └── split/
        ├── train.pkl                     # List[video_id]
        ├── valid.pkl
        └── test.pkl

kaggle-working/
├── dinov3_T16_dim1024_merge/             # 🆕 Merged DINOv3 features (all splits)
│   ├── video_id_1.pt
│   ├── video_id_2.pt
│   └── ...  (~16K files)
│
└── models/
    └── best_model_dinov3_ncod.ckpt       # Checkpoint: model + U
```

### 4.2 Feature Specifications

| Feature | Shape | Source | File Format |
|---------|-------|--------|-------------|
| DINOv3 frame | `[T, 1024]` | DINOv3 ViT-L/14 [CLS] | `.pt` (per video) |
| Object | `[T, O, 2053]` | DETR features + bbox | `.pkl` (per video) |
| Text | tokenized on-the-fly | DeBERTa tokenizer | N/A |

### 4.3 Text Annotation Format

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

## 5. End-to-End Pipeline

### 5.1 Data Preprocessing

1. **DINOv3 Feature Extraction** (offline):
   - Extract per-frame [CLS] tokens using DINOv3 ViT-L/14.
   - Save as `{video_id}.pt` with shape `[T, 1024]`.
   - Merge train/valid/test splits into single directory.

2. **Object Feature** (offline, sẵn có):
   - DETR detects objects per frame → features `[O, 2048]` + bboxes `[O, 4]`.
   - Transform bboxes via `transform_bb(bbox, W=640, H=480)` → `[O, 5]`.
   - Concat → `[O, 2053]` per frame.

### 5.2 Dataset Construction (`VideoQADataset` in `DataLoader.py`)

- Load video IDs from split `.pkl` files.
- **Scan** DINOv3 `.pt` files và object `.pkl` files → build index maps (O(1) lookup).
- Filter videos: chỉ giữ video có **cả** DINOv3 features VÀ object features.
- Parse annotations: generate QA pairs cho all 6 question types.
- `__getitem__` trả thêm `idx` (sample index) — **cần thiết cho NCOD** để index vào `U[idx]`.

```python
# Return format (real-time text encoding mode):
return ff, of, qns, ans_word, ans_id, qns_key, idx
#      ↑    ↑   ↑      ↑        ↑       ↑       ↑
#    [T,1024] [T,O,2053] str  list[str]  int    str   int
```

### 5.3 Model Forward Pass

```
Input: ff [B,T,1024], of [B,T,O,2053], qns (str), ans (list[str])
                    │
    ┌───────────────┼───────────────┐──────────────────────┐
    ▼               ▼               ▼                      ▼
FeatureResizer  ObjResize      forward_text          forward_text
1024→768        2053→768       (question)            (5 answers)
    │               │               │                      │
    ▼               │               ▼                      │
FrameDecoder ◄──────┼──────── q_local ──────────►  ObjDecoder
    │               │               │                      │
 frame_att          │               │                   obj_att
    │               │               │                      │
 TopK(5 frames)     │               │               TopK(12 objs)
    │               │               │                      │
    ▼               ▼               │                      ▼
 selected_frames  selected_objs    │              selected_objs
    │               │               │                      │
    └───────────────┘               │                      │
            │                       │                      │
    FrameObjDecoder                 │                      │
            │                       │                      │
    VLEncoder ◄─────────────────────┘                      │
            │                                              │
    AnsDecoder ◄───────────────────────────────────────────┘
            │                                    ([CLS] of each QA pair)
    Classifier                     
            │
    logits [B, 5]
```

### 5.4 NCOD Bi-level Training

```python
# Pseudo-code for train_epoch_ncod()
for batch in loader:
    ff, of, q, a, ans_id, _, sample_indices = batch
    
    logits = model(ff, of, q, a)           # Forward
    probs = softmax(logits)                 # [B, 5]
    
    u = U[sample_indices].unsqueeze(1)      # [B, 1] — per-sample discount
    y = one_hot(ans_id, 5)                  # [B, 5]
    
    # L1: update model (U frozen)
    shifted = probs + u.detach() * y        # Shift prediction toward label
    L1 = -mean(sum(y * log(clamp(shifted))))
    (L1 / accum_steps).backward()           # Accumulate gradients
    
    # L2: update U (model frozen)
    shifted_u = probs.detach() + u * y
    L2 = MSE(shifted_u, y)
    (L2 / accum_steps).backward()
    
    if step % accum_steps == 0:
        clip_grad_norm_(model.parameters())
        opt_model.step(); opt_U.step()
        U.clamp_(0, 0.99)
```

### 5.5 Evaluation

- **Không dùng U** khi evaluation — model chạy forward pass bình thường, argmax logits.
- Best checkpoint (highest val_acc) được reload trước evaluation.
- Metrics tính trên validation set.

---

## 6. Cài đặt & Sử dụng

### 6.1 Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 1.11+
- transformers (DeBERTa)
- h5py
- einops
- wandb
- tqdm, seaborn, matplotlib

### 6.2 Download dữ liệu

```python
import kagglehub

# DINOv3 features
dinov3_path = kagglehub.dataset_download('your-username/dinov3-feat')

# Object features
obj_path = kagglehub.dataset_download('your-username/object-detection-causal-full')

# Text annotations
text_path = kagglehub.dataset_download('lusnaw/text-annotation')

# Data splits
split_path = kagglehub.dataset_download('your-username/casual-vid-data-split')
```

### 6.3 Training (Kaggle Notebook)

Sử dụng `train-transtr-different-feature_ncod_ga.ipynb`:

| Cell | Nội dung |
|------|----------|
| 0-1 | Git clone, setup |
| 2 | W&B login |
| 3 | Imports |
| 4 | Train/Eval functions (NCOD bi-level + gradient accumulation) |
| 5 | Merge DINOv3 features (train/valid/test → merged) |
| 6 | Config & paths |
| 7 | Create datasets & dataloaders |
| 8 | Model + NCOD U initialization + dual optimizers |
| 9 | W&B run initialization |
| 10 | Training loop with NCOD + W&B logging |
| 11 | Detailed evaluation + charts |
| 12 | W&B finish |
| 13 | Noisy sample analysis + export |

### 6.4 Monitoring trên W&B

#### U Histogram Interpretation

| Phân phối U | Ý nghĩa |
|-------------|---------|
| Tất cả U ≈ 0 (unimodal) | U chưa phân biệt được → cần thêm epochs |
| Bimodal: đỉnh ≈ 0 + đuôi dài 0.3–0.9 | ✅ **Thành công** — đuôi dài = suspected noisy |
| U tăng đều toàn bộ | L₂ loss có vấn đề, kiểm tra gradient |

#### Metrics được log

```
Epoch-level:
  train_L1, train_L2, train_acc, val_acc
  U/mean, U/std, U/max, U/min
  U/pct_gt_0.5, U/pct_gt_0.9
  U/histogram (wandb.Histogram)

Batch-level (mỗi 50 batches):
  batch_L1, batch_L2, batch_acc

Evaluation:
  eval/Description, eval/Explanation
  eval/Predictive_Answer, eval/Predictive_Reason
  eval/Counterfactual_Answer, eval/Counterfactual_Reason
  eval/PAR, eval/CAR, eval/Acc_ALL
```

---

## 7. Checkpoint Format

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'U': U.detach().cpu(),           # [N_train] — discount parameters
    'epoch': best_epoch,
    'val_acc': best_val_acc,
}
torch.save(checkpoint, 'best_model_dinov3_ncod.ckpt')
```

Checkpoint được tự động upload lên W&B dưới dạng artifact `best-model-dinov3-ncod`.

---

## 8. Noisy Sample Analysis

Sau training, Cell 13 phân tích phân phối U:

1. **Export CSV đầy đủ**: `ncod_all_video_U.csv` chứa `(sample_idx, qns_key, U_value)` cho tất cả training samples.
2. **Top-K noisy samples**: `ncod_noisy_samples.csv` — 100 samples có U cao nhất.
3. **W&B Table**: `ncod_video_id_U_table` — viewable trực tiếp trên W&B dashboard.
4. **U Distribution plot**: Histogram toàn bộ + histogram filtered (U > 0.01).

### Ví dụ output:
```
📊 U Distribution (all 48000 samples):
  mean=2.3456e-03  std=4.5678e-02
  min=0.0000e+00   max=9.8765e-01
  >0.5: 127 (0.3%)
  >0.9: 23 (0.0%)

Top-5 noisiest samples:
#1 | U=0.9877 | vid=video_1234 | type=counterfactual_reason
#2 | U=0.9654 | vid=video_5678 | type=explanatory
#3 | U=0.9432 | vid=video_9012 | type=predictive_reason
...
```

---

## 9. So sánh các thay đổi kiến trúc

### 9.1 Phân tích chuyên sâu: TranSTR (Baseline) vs TranSTR-DN (Ours)

Sự khác biệt cốt lõi giữa kiến trúc hiện tại (TranSTR-DN) và TranSTR gốc (CVPR 2022) nằm ở 3 khía cạnh: **Biểu diễn hình ảnh (Visual Representation)**, **Cơ chế chống nhiễu (Noise-Aware Learning)**, và **Tối ưu hóa huấn luyện (Training Optimization)**. Kiến trúc mạng Transformer (Encoder/Decoder) được giữ nguyên để chứng minh hiệu quả thuần túy của các thay đổi này.

#### 1. Sự Dịch Chuyển Về Visual Backbone
| Đặc điểm | TranSTR gốc (ResNet Duality) | TranSTR-DN hiện tại (DINOv3) | Ý nghĩa thực tiễn của sự thay đổi |
| :--- | :--- | :--- | :--- |
| **Mô hình cốt lõi** | ResNet-101 (ImageNet) + 3D-ResNet | **DINOv3 ViT-L/14** (LVD-142M) | Chuyển từ supervised learning sang **self-supervised learning**. DINOv3 học ngữ nghĩa toàn cục (scene semantics) thay vì chỉ học phân loại object/action cụ thể. |
| **Độ đo đặc trưng (Dim)**| `4096` = Concat(`2048` app + `2048` mot) | **`1024`** (`[CLS]` token) | Giảm 4 lần số lượng tham số đầu vào cho module `FeatureResizer`. Điều này giúp mạng tránh bị overfit vào nhiễu cục bộ và giảm chi phí tính toán. |
| **Sự liên kết Không-Thời gian**| Trộn lẫn thủ công qua hàm `Concat` | **Tự nhiên** thông qua Attention Map | TranSTR gốc dùng 3D-ResNet cho motion, tuy nhiên motion không phải lúc nào cũng giải thích được "Nguyên nhân" (Explanatory). DINOv3 sinh ra attention map khu vực tốt hơn, giúp module `Frame Decoder` chọn ra các khung hình thực sự chứa *mấu chốt nhân quả*. |

#### 2. Đột Phá Ở Hàm Mất Mát: NCOD (Noisy Correspondence Detection)
Đây là sự lột xác lớn nhất về thiết kế hàm mục tiêu (objective function).

*   **TranSTR gốc (Niềm tin tuyệt đối)**: Sử dụng hàm `Cross-Entropy` cơ bản. Mạng luôn "tin" Ground Truth là chính xác 100%. Tuy nhiên, ở bộ dữ liệu CausalVidQA, các câu hỏi như "Tại sao người đàn ông lại chạy?" có tính đa nghĩa cao. Nhãn A có thể hợp lý với người này, nhãn B hợp lý với người khác. Việc ép mô hình học 1 nhãn duy nhất đẩy mạng vào trạng thái **Overfit Noise** (nhớ vẹt nhãn sai thay vì học quy luật).
*   **TranSTR-DN (Học có chọn lọc)**: Triển khai chiến lược **Bi-level Optimization**.
    *   Thêm một vector tham số chiết khấu `U` (độ dài bằng số lượng video train).
    *   **Level 1 (Cập nhật Model - L1):** `Loss = CE(softmax + U.detach() * Y_true, Y_true)`. Nếu câu hỏi bị nghi ngờ là nhiễu, `U` sẽ tự động tăng cao. Xác suất tại vị trí Ground Truth được "nâng đỡ" bù vào, dẫn đến `Loss` tự động nhỏ lại -> **Mô hình bỏ qua học mẫu này**.
    *   **Level 2 (Cập nhật U - L2):** `Loss = MSE(softmax.detach() + U * Y_true, Y_true)`. Nếu mô hình (sau khi đã detach gradient) vẫn dự đoán sai khác nhiều so với Ground Truth, tham số `U` sẽ tăng lên (bị phạt), gắn cờ đây là "Mẫu Khó / Cực Nhiễu".

#### 3. Cải Tiến Về Tối Ưu Hóa Kỹ Thuật (Optimization Engineering)
*   **Hệ thống Dual Optimizers**: TranSTR gốc dùng 1 Optimizer. TranSTR-DN dùng **AdamW** (lr=1e-5) cho Neural Network và **SGD** (lr=0.1) tịnh tiến riêng cho Vector `U`. Việc tách biệt giúp Vector `U` phản ứng cực nhanh (learning rate cao) với các nhãn có dấu hiệu nhiễu, ngay từ những epoch đầu tiên.
*   **Gradient Accumulation**: Các bài báo gốc chạy batch size lớn (16-32) yêu cầu VRAM rât lớn (đặc biệt khi load text qua mô hình DeBERTa). Trong TranSTR-DN, chúng tôi thiết kế lại `train_epoch_ncod` dùng `bs=8` kết hợp `accumulation_steps=4`. Kết quả: **Effective Batch Size = 32** nhưng chỉ ăn VRAM của `bs=8`.
*   **Regularization Rate**: TranSTR gốc dùng dropout = 0.1. Mạng DINOv3 có tính biểu diễn (representation capacity) quá rực rỡ, nếu không kìm hãm mạng sẽ bay nhanh vào trạng thái ghi nhớ (memorization). Nên chúng tôi điều chỉnh dropout hiện tại = **0.3**.
*   **Truy xuất & Lưu trữ State (Resume-ability)**: Bản cũ chỉ lưu trọng số (`model.state_dict()`). Bản TranSTR-DN lưu một **Full Checkpoint Object** chứa Model, `U`, AdamW_state, SGD_state, Scheduler_state, History và Best_Acc. Giúp khả năng tạm dừng và Resume training bảo toàn trọn vẹn adaptive learning rate. Tối quan trọng khi thời gian train dài.

---

## 10. Known Issues & Solutions

### 1. DeBERTa FP16 Overflow
**Vấn đề**: `RuntimeError: value cannot be converted to type at::Half`  
**Giải pháp**: Tắt mixed precision — `USE_AMP = False`

### 2. Multiprocessing Error trên Kaggle
**Vấn đề**: `Bad file descriptor` với `num_workers > 0`  
**Giải pháp**: `DataLoader(..., num_workers=0)` hoặc test với `num_workers=4`

### 3. U tăng quá nhanh
**Vấn đề**: Tất cả U tăng đều → model không học  
**Giải pháp**: Giảm `ncod_u_lr` (thử 0.01 thay vì 0.1)

### 4. U không tăng sau 10 epoch
**Vấn đề**: U histogram vẫn unimodal gần 0  
**Giải pháp**: Tăng `ncod_u_lr` hoặc kiểm tra L₂ gradient flow

### 5. Train loss NaN
**Vấn đề**: `shifted_probs` quá nhỏ gây log(0)  
**Giải pháp**: `torch.clamp(shifted_probs, min=1e-12, max=1.0)` (đã implement)

---

## 11. References

- [IGV — Invariant Grounding for Video Question Answering (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
- [CausalVidQA Dataset](https://github.com/bcmi/Causal-VidQA)
- [Original IGV Code](https://github.com/yl3800/IGV)
- [DINOv2/v3 — Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [NCOD — Learning with Noisy Correspondence for Cross-modal Matching](https://arxiv.org/abs/2304.13756)

---

## 12. Citation

```bibtex
@InProceedings{Li_2022_CVPR,
    author    = {Li, Yicong and Wang, Xiang and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title     = {Invariant Grounding for Video Question Answering},
    booktitle = {CVPR},
    year      = {2022},
    pages     = {2928-2937}
}

@article{oquab2024dinov2,
    title     = {DINOv2: Learning Robust Visual Features without Supervision},
    author    = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
    journal   = {Transactions on Machine Learning Research},
    year      = {2024}
}
```