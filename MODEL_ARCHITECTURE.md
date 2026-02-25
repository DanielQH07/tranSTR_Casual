# Model Architecture - TranSTR + TokenMark (SoM)

## Tổng quan

**TranSTR** (Transformer-based VideoQA) với **TokenMark (Set-of-Mark) injection** cho dataset CausalVidQA.

- **Task**: Multiple-choice VideoQA (5 lựa chọn)
- **Dataset**: CausalVidQA
- **Evaluation**: Description, Explanation, PAR, CAR, Acc_ALL

---

## Sơ đồ tổng thể

```
Frame features [B, 16, 1024]  ──► frame_resize ──► [B, 16, 768]
                                                          │
                                                   frame_decoder ◄── q_local [B, seq, 768]
                                                          │ frame_att
                                                    PerturbedTopK
                                                          │ idx_frame
                                          ┌───────────────┴───────────────┐
                                   frame_local_raw                  obj_feat_selected
                                   [B, topK_f, 768]        [B, topK_f, 20, 2053]
                                          │                        │ obj_resize
                                          │                  obj_local_raw [B, topK_f, 20, 768]
                                          │                        │
                                          └──── SoMInjector ───────┘  (detached, no_grad)
                                                          │ idx_obj (SoM-guided selection)
                                                          │
                                                   obj_decoder(obj_local_raw) ◄── q_local_repeated
                                                          │ topK obj selection via idx_obj
                                                   obj_local [B, topK_f, topK_o, 768]
                                                          │
                                              frame_local = frame_local_raw [B, topK_f, 768]
                                                          │
                                                    fo_decoder  (frame ↔ obj cross-attn)
                                                          │ frame_obj [B, topK_f, 768]
                                                          │
                                                    vl_encoder  [frame_obj ∥ q_local]
                                                          │ mem [B, topK_f + seq, 768]
                                                          │
Object text [B*5 Q-A pairs] ──► forward_text ──► tgt [CLS token, B, 5, 768]
                                                          │
                                                   ans_decoder(tgt, mem)
                                                          │
                                                    classifier Linear(768→1)
                                                          │
                                                    logits [B, 5]
```

---

## Components

### 1. Text Encoder (`forward_text`)
| Item | Value |
|------|-------|
| Model | `microsoft/deberta-base` |
| Output | `last_hidden_state` → `[B, seq_len, 768]` |
| Projection | `Linear(768 → d_model=768)` |
| Freeze | `False` (trainable) |

Dùng cho cả **question** (`qns_word`) và **answer candidates** (`ans_word`).  
Answer encoding: `[CLS]` token của mỗi cặp `[Q SEP Ai]` làm query cho `ans_decoder`.

---

### 2. Feature Resizers
| Resizer | Input dim | Output dim |
|---------|-----------|------------|
| `frame_resize` | 1024 (ViT) | 768 (d_model) |
| `obj_resize` | 2053 (ROI feat 2048 + bbox 5) | 768 (d_model) |

Cấu trúc: `Linear → LayerNorm → Dropout`

---

### 3. Hierarchical Transformer Decoders

#### 3a. Frame Decoder (`frame_decoder`)
```
Input (query):  frame_feat  [B, 16, 768]  + pos_encoding
Memory:         q_local     [B, seq, 768]
Output:         frame_local [B, 16, 768]  + frame_att [B, 16, seq]
```
→ `frame_att` đưa vào `PerturbedTopK` → `idx_frame` → chọn `topK_frame=5` frames

#### 3b. Object Decoder (`obj_decoder`) — gọi 2 lần với mục đích khác nhau
```
Pass 1 (SoM-guided, detached, no_grad):
  Input:  obj_local_som  [B*topK_f, 20, 768]  (SoM-injected, detached)
  Output: obj_att_som    [B*topK_f, 20, seq]
  → PerturbedTopK → idx_obj  (SoM-guided object selection index)

Pass 2 (raw features, có grad):
  Input:  obj_local_raw  [B*topK_f, 20, 768]  (raw, không có SoM)
  Output: obj_local_flat [B*topK_f, 20, 768]
  → apply idx_obj → obj_local [B, topK_f, topK_o=12, 768]
```

#### 3c. Frame-Object Decoder (`fo_decoder`)
```
Query:   frame_local  [B, 5, 768]        (raw, không SoM)
Memory:  obj_local    [B, 5*12, 768]     (raw, không SoM)
Output:  frame_obj    [B, 5, 768]
```

---

### 4. Vision-Language Encoder (`vl_encoder`)
```
Input:  [frame_obj ∥ q_local]  [B, 5+seq_len, 768]
Output: mem                    [B, 5+seq_len, 768]
```
Self-attention encoder kết hợp visual và text context.

---

### 5. Answer Decoder (`ans_decoder`)
```
Query:   tgt = a_seq[:,:,0,:]   [B, 5, 768]   ([CLS] của mỗi answer candidate)
Memory:  mem                     [B, 5+seq, 768]
Output:  out                     [B, 5, 768]
→ classifier Linear(768→1) → squeeze → logits [B, 5]
```

---

### 6. TokenMark (SoM) Injector (`som_injector`)

**Vai trò**: Chỉ dùng để **guide object selection** (tính `idx_obj`). **Không ảnh hưởng** đến features đưa vào `fo_decoder`, `vl_encoder`, `ans_decoder`.

```
SoMInjector
├── TokenMarkPalette:  Embedding(num_marks=16, d_model=768)
├── proj_frame:        Linear(768→768) + LayerNorm
├── proj_obj:          Linear(768→768) + LayerNorm
├── gamma_frame:       learnable scalar (init=0.1)
└── gamma_obj:         learnable scalar (init=0.1)
```

**Input**: `frame_local_raw`, `obj_local_raw`, `som_data` (frame masks + entity names)  
**Output**: `frame_local_som`, `obj_local_som` — chỉ dùng trong `torch.no_grad()` block để tính attention score

---

### 7. TopK Selection

| Selector | Training | Inference (hard_eval=False) | Inference (hard_eval=True) |
|----------|----------|-----------------------------|---------------------------|
| `frame_sorter` | `PerturbedTopK(5)` | `PerturbedTopK(5)` | `HardtopK(5)` |
| `obj_sorter` | `PerturbedTopK(12)` | `PerturbedTopK(12)` | `HardtopK(12)` |

---

## Hyperparameters

```python
# Architecture
d_model          = 768
nheads           = 8
num_encoder_layers = 2      # dùng cho cả encoder lẫn decoder layers
activation       = 'gelu'
normalize_before = True
dropout          = 0.3
encoder_dropout  = 0.3

# Features
frame_feat_dim   = 1024     # ViT
obj_feat_dim     = 2053     # ROI (2048 + 5 bbox)
frames           = 16       # Tổng frames load
objs             = 20       # Max objects/frame
topK_frame       = 5        # Frames sau selection
topK_obj         = 12       # Objects sau selection

# SoM
use_som          = True     # nếu có dữ liệu mask
num_marks        = 16

# Training
batch_size       = 8
learning_rate    = 1e-5
weight_decay     = 1e-4
epochs           = 20
patience         = 5        # ReduceLROnPlateau
lr_gamma         = 0.1
```

---

## Tensor shapes qua các stage

```
Stage                          Shape
─────────────────────────────────────────────────────────
frame_feat (sau resize)        [B, 16, 768]
q_local                        [B, seq_len, 768]
frame_local (frame_decoder)    [B, 16, 768]
idx_frame                      [B, 16, topK_f=5]
frame_local_raw                [B, 5, 768]
obj_feat_selected              [B, 5, 20, 2053]
obj_local_raw (sau resize)     [B, 5, 20, 768]
obj_local_som (SoM-injected)   [B, 5, 20, 768]   ← detached
idx_obj                        [B*5, 20, topK_o=12]
obj_local (sau selection)      [B, 5, 12, 768]
frame_obj (fo_decoder)         [B, 5, 768]
mem (vl_encoder)               [B, 5+seq_len, 768]
tgt ([CLS] answer)             [B, 5, 768]
out (ans_decoder)              [B, 5, 768]
logits (classifier)            [B, 5]
```

---

## Model Size (ước tính)

| Component | Params |
|-----------|--------|
| DeBERTa-base | ~86M |
| Transformer layers (5 × decoder/encoder) | ~20M |
| frame_resize + obj_resize | ~2M |
| SoM injector (palette + proj) | ~1.2M |
| classifier | ~4K |
| **Total** | **~109M** |

---

## References

- DeBERTa: https://huggingface.co/microsoft/deberta-base
- CausalVidQA dataset
- TokenMark (SoM): Set-of-Mark prompting for visual grounding
