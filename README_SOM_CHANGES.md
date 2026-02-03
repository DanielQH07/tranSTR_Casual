# Token Mark (SoM) Integration for TranSTR

## Tổng quan

Tích hợp **Set-of-Mark Token Injection** vào TranSTR cho **explicit entity grounding** trong Causal-VidQA.

Token Marks đóng vai trò như **learnable causal anchors** để kết nối các entities trong câu hỏi với các vùng tương ứng trong video frames.

---

## Full Flow - Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LOADING                             │
├─────────────────────────────────────────────────────────────────┤
│  VideoQADataset.__getitem__(idx)                                │
│  ├── Load ViT features: [16, 1024]                              │
│  ├── Load Object features: [16, 20, 2053]                       │
│  ├── Load Question/Answers text                                 │
│  └── Load SoM data: _load_som_features(vid)                     │
│      ├── id_masks/<vid>.npz → frame_masks {0: [H,W], ...}       │
│      └── metadata_json/<vid>.json → entity_names {1: "person"}  │
│                                                                 │
│  Return: (ff, of, qns, ans, ans_id, qns_key, som_data)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      COLLATE FUNCTION                           │
├─────────────────────────────────────────────────────────────────┤
│  collate_fn_som(batch):                                         │
│  ├── ff: [B, 16, 1024] - stacked frame features                 │
│  ├── of: [B, 16, 20, 2053] - stacked object features            │
│  ├── qns: List[str] - questions                                 │
│  ├── ans: List[List[str]] - answer options                      │
│  ├── ans_id: [B] - correct answer indices                       │
│  ├── qns_key: List[str] - question keys                         │
│  └── som_data: List[Dict] - SoM data per sample                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL FORWARD PASS                          │
│                    (VideoQAmodel.forward)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. RESIZE FRAME FEATURES                                       │
│     frame_feat = frame_resize(frame_feat)                       │
│     [B, 16, 1024] → [B, 16, 768]                                 │
│                                                                 │
│  2. ENCODE QUESTION                                             │
│     q_local, q_mask = forward_text(qns)                         │
│     → [B, seq_len, 768]                                         │
│                                                                 │
│  3. FRAME DECODER + TOPK SELECTION                              │
│     frame_local, frame_att = frame_decoder(frame_feat, q_local) │
│     idx_frame = frame_sorter(frame_att)                         │
│     frame_local = weighted_select(frame_local, idx_frame)       │
│     [B, 16, 768] → [B, 4, 768]  (topK_frame=4)                   │
│                                                                 │
│  4. RESIZE OBJECT FEATURES                                      │
│     obj_feat = weighted_select(obj_feat, idx_frame)             │
│     obj_local = obj_resize(obj_feat)                            │
│     [B, 4, 20, 2053] → [B, 4, 20, 768]                           │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  5. SoM INJECTION (if use_som=True)                       ║  │
│  ║     frame_local, obj_local = som_injector(                ║  │
│  ║         frame_local,    # [B, 4, 768]                     ║  │
│  ║         obj_local,      # [B, 4, 20, 768]                 ║  │
│  ║         som_data,       # List[Dict]                      ║  │
│  ║         idx_frame       # [B, 16, 4] frame selection      ║  │
│  ║     )                                                     ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│  6. OBJECT DECODER + TOPK SELECTION                             │
│     obj_local = obj_decoder(obj_local, q_local)                 │
│     [B, 4, 20, 768] → [B, 4, 5, 768]  (topK_obj=5)               │
│                                                                 │
│  7. HIERARCHY GROUPING                                          │
│     frame_obj = fo_decoder(frame_local, obj_local)              │
│                                                                 │
│  8. VL FUSION + ANSWER DECODING                                 │
│     mem = vl_encoder(frame_obj + q_local)                       │
│     out = ans_decoder(tgt, mem)                                 │
│     out = classifier(out)  → [B, 5]                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## SoM Injection Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    SoMInjector.forward()                        │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                         │
│  ├── frame_local: [B, frame_topK, d_model]                      │
│  ├── obj_local: [B, frame_topK, O, d_model]                     │
│  ├── som_data: List[Dict] per batch                             │
│  └── idx_frame: [B, F, frame_topK] selection weights            │
│                                                                 │
│  For each batch item b:                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Parse SoM Data                                        │  │
│  │     frame_masks = som_data[b]['frame_masks']              │  │
│  │     entity_names = som_data[b]['entity_names']            │  │
│  │                                                           │  │
│  │  2. Get Active Mark Embeddings                            │  │
│  │     K = max(entity_names.keys())                          │  │
│  │     mark_emb = palette(0..K-1)  → [K, 768]                │  │
│  │                                                           │  │
│  │  3. Map Frame Indices                                     │  │
│  │     Use idx_frame to find which original frames           │  │
│  │     contribute most to each selected frame                │  │
│  │     top_orig = idx_frame[b].argmax(dim=0)                 │  │
│  │     mapped_masks = {sel_idx: masks[orig_idx]}             │  │
│  │                                                           │  │
│  │  4. Inject Frame Features                                 │  │
│  │     For each selected frame with mask:                    │  │
│  │     ├── Count pixels per entity (normalized)              │  │
│  │     ├── Weighted sum of projected marks                   │  │
│  │     └── frame[t] += gamma * spatial_mark                  │  │
│  │                                                           │  │
│  │  5. Inject Object Features                                │  │
│  │     For each frame:                                       │  │
│  │     ├── Average marks of entities present                 │  │
│  │     └── objects[t] += gamma * avg_mark                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Output:                                                        │
│  ├── injected_frames: [B, frame_topK, d_model]                  │
│  └── injected_objs: [B, frame_topK, O, d_model]                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Format

### Directory Structure
```
obj_mask_causal_full/
├── id_masks/
│   ├── video_001.npz
│   ├── video_002.npz
│   └── ...
│
└── metadata_json/
    ├── video_001.json
    ├── video_002.json
    └── ...
```

### NPZ File Format (`id_masks/<video_id>.npz`)
```python
# Keys: f0, f1, f2, ..., f15 (16 frames)
# Values: (H, W) uint8 array
#   - 0 = background
#   - 1, 2, 3, ... = entity IDs

masks = np.load('video_001.npz')
print(masks.files)  # ['f0', 'f1', ..., 'f15']
print(masks['f0'].shape)  # (480, 640) or similar
print(np.unique(masks['f0']))  # [0, 1, 2] = background + 2 entities
```

### JSON File Format (`metadata_json/<video_id>.json`)
```json
{
  "id_to_label": {
    "1": "person_1",
    "2": "car_1",
    "3": "person_2"
  }
}
```

---

## Key Components

### 1. `TokenMarkPalette`
Learnable embedding table cho Token Marks.

```python
palette = TokenMarkPalette(num_marks=16, d_model=768)
# palette.marks: nn.Embedding(16, 768)
# marks[k] = learnable vector for entity k
```

### 2. `SoMInjector`
Main module tích hợp vào VideoQAmodel.

```python
som_injector = SoMInjector(
    d_model=768,      # Feature dimension
    obj_feat_dim=768, # Same as d_model after resize
    num_marks=16,     # Max number of entities
    gamma_init=0.1,   # Initial injection scale
)
```

**Learnable parameters:**
- `palette.marks`: 16 × 768 mark embeddings
- `proj_frame`: Linear(768→768) + LayerNorm
- `proj_obj`: Linear(768→768) + LayerNorm
- `gamma_frame`: scalar injection scale for frames
- `gamma_obj`: scalar injection scale for objects

### 3. Frame Injection Equation

```
Eq. 4.3: S_t = Σ_k (w_k · P_frame(r_k))

where:
  w_k = pixels_of_entity_k / total_entity_pixels  (normalized)
  r_k = palette[k]  (mark embedding)
  P_frame = proj_frame (projection layer)
  
frame_local[t] += gamma_frame * S_t
```

### 4. Object Injection Equation

```
Eq. 4.2: M_t = (1/N) · Σ_k P_obj(r_k)

where:
  N = number of entities in frame t
  r_k = palette[k]
  P_obj = proj_obj

obj_local[t, :, :] += gamma_obj * M_t  (broadcast to all objects)
```

---

## Files Modified

### `networks/model.py`
```python
# In __init__:
self.use_som = use_som
if use_som:
    self.som_injector = SoMInjector(d_model=self.d_model, ...)

# In forward:
# AFTER frame/object resize and topK selection:
if self.use_som and som_data is not None:
    frame_local, obj_local = self.som_injector(
        frame_local, obj_local, som_data, idx_frame=idx_frame
    )
```

### `DataLoader.py`
```python
# In __init__:
self.som_feature_path = som_feature_path

# In __getitem__:
som_data = self._load_som_features(vid)
return ff, of, qns, ans_word, ans_id, qns_key, som_data

# New method:
def _load_som_features(self, vid):
    # Load from id_masks/<vid>.npz and metadata_json/<vid>.json
    ...
```

### `networks/som_injection.py`
New file containing:
- `TokenMarkPalette`
- `VisualMarkInjector` (legacy, unused)
- `TextMarkInjector` (legacy, unused)
- `EntityMatcher`
- `SoMInjector` (main v2)

---

## Usage

### Training with SoM
```python
# 1. Config paths
SOM_FEATURE_PATH = '/kaggle/input/obj-mask-causal'

# 2. Create dataset
train_ds = VideoQADataset(
    split='train',
    ...,
    som_feature_path=SOM_FEATURE_PATH
)

# 3. Create model with SoM enabled
model = VideoQAmodel(
    ...,
    use_som=True,
    num_marks=16
)

# 4. Custom collate function
def collate_fn_som(batch):
    ff = torch.stack([item[0] for item in batch])
    of = torch.stack([item[1] for item in batch])
    qns = [item[2] for item in batch]
    ans = [item[3] for item in batch]
    ans_id = torch.tensor([item[4] for item in batch])
    qns_key = [item[5] for item in batch]
    som_data = [item[6] for item in batch]  # List of dicts
    return ff, of, qns, ans, ans_id, qns_key, som_data

loader = DataLoader(train_ds, batch_size=8, collate_fn=collate_fn_som)

# 5. Training loop
for batch in loader:
    ff, of, qns, ans, ans_id, qns_key, som_data = batch
    ff, of = ff.to(device), of.to(device)
    
    out = model(ff, of, qns, ans, som_data=som_data)
    loss = criterion(out, ans_id.to(device))
    ...
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    out = model(ff, of, qns, ans, som_data=som_data)
    preds = out.argmax(dim=-1)
```

---

## Bug Fixes (v2 - 2026-01-22)

### ❌ Bug 1: Injection timing sai
**Trước:** SoM injection xảy ra TRƯỚC `obj_resize` → thông tin inject bị mất khi resize.
**Sau:** Injection xảy ra SAU khi cả frame và object đã resize về d_model.

### ❌ Bug 2: Mask weights không normalize
**Trước:** `mask_sum` (hàng chục ngàn pixels) dùng trực tiếp → gradient explosion.
**Sau:** Weights được normalize: `w_k = mask_sum / total_entity_pixels`.

### ❌ Bug 3: Frame index không khớp
**Trước:** Masks cho original frames (0-15) được apply 1:1 vào selected frames.
**Sau:** Dùng `idx_frame` để map từ selected frames về original frames.

### ❌ Bug 4: `object_to_entity` rỗng
**Trước:** Object injection phụ thuộc vào mapping không có trong data.
**Sau:** Inject average entity marks vào tất cả objects trong frame.

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-22 | **v2** | Fixed injection timing, mask normalization, frame mapping |
| 2026-01-21 | v1.1 | Fixed data format (f0 keys, id_to_label) |
| 2026-01-21 | v1.0 | Initial Token Mark integration |

---

## Troubleshooting

### Model accuracy very low (~12-20%)
- Check if SoM injection is happening at the right time (after resize)
- Verify mask weights are normalized
- Ensure `idx_frame` is passed correctly

### CUDA out of memory
- Reduce batch size
- Use `num_workers=0` in DataLoader
- Add `torch.cuda.empty_cache()` periodically

### SoM data is None for some videos
- Not all videos may have SoM annotations
- Code handles this gracefully (no injection if None)

---

## References

- Set-of-Mark Visual Prompting: https://arxiv.org/abs/2310.11441
- TranSTR paper: [link]
- Causal-VidQA dataset: [link]
