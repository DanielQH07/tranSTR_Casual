# Token Mark (SoM) Integration - Changelog

## Tổng quan
Tích hợp **Set-of-Mark Token Injection** vào TranSTR cho explicit entity grounding trong Causal-VidQA.

---

## Files đã tạo mới

### `networks/som_injection.py`
Module chính cho Token Mark injection:
- `TokenMarkPalette` - Learnable mark embeddings (16 tokens, 768-dim)
- `VisualMarkInjector` - Inject marks vào frame/object features
- `TextMarkInjector` - Inject marks vào text embeddings
- `SoMInjector` - Main orchestrator module

---

## Files đã sửa

### `networks/model.py`
```diff
+ from networks.som_injection import SoMInjector

  class VideoQAmodel:
-     def __init__(self, ..., **kwargs):
+     def __init__(self, ..., use_som=False, num_marks=16, **kwargs):
+         if use_som:
+             self.som_injector = SoMInjector(...)

-     def forward(self, frame_feat, obj_feat, qns_word, ans_word):
+     def forward(self, frame_feat, obj_feat, qns_word, ans_word, som_data=None):
+         if self.use_som and som_data is not None:
+             frame_feat, obj_feat = self.som_injector(frame_feat, obj_feat, som_data)
```

### `DataLoader.py`
```diff
  class VideoQADataset:
-     def __init__(self, ..., text_feature_path=None):
+     def __init__(self, ..., text_feature_path=None, som_feature_path=None):
+         self.som_feature_path = som_feature_path

      def __getitem__(self, idx):
          ...
+         som_data = self._load_som_features(vid)
-         return ff, of, qns, ans_word, ans_id, qns_key
+         return ff, of, qns, ans_word, ans_id, qns_key, som_data

+     def _load_som_features(self, vid):
+         # Load from id_masks/<vid>.npz và metadata_json/<vid>.json
+         # NPZ keys: f0, f1, ..., f15 (H×W mask)
+         # JSON: {"id_to_label": {"1": "person_1", ...}}
```

### `train.py`
```diff
  def train(...):
-     vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, _ = inputs
+     vid_frame_inputs, vid_obj_inputs, qns_w, ans_w, ans_id, _, som_data = inputs
```

### `transtr_wandb.ipynb`
- Thêm `SOM_FEATURE_PATH` config
- Thêm `collate_fn_som` để handle som_data list
- Thêm CELL 6.5: Verify SoM Data
- Update train/eval functions với `use_som` flag

---

## Data Format

```
obj_mask_causal_full/
├── id_masks/<video_id>.npz
│   Keys: f0, f1, ..., f15
│   Values: (H, W) uint8 mask, 0=background, 1..N=entity IDs
│
└── metadata_json/<video_id>.json
    {"id_to_label": {"1": "person_1", "2": "person_2", ...}}
```

---

## Cách sử dụng

```python
# 1. Config path
SOM_FEATURE_PATH = '/kaggle/input/causal-vqa-object-masks-full/obj_mask_causal_full'

# 2. Dataset
train_ds = VideoQADataset(..., som_feature_path=SOM_FEATURE_PATH)

# 3. Model
model = VideoQAmodel(..., use_som=True, num_marks=16)

# 4. Forward
out = model(ff, of, qns, ans, som_data=som_data)
```

---

## Ngày cập nhật
- 2026-01-21: Initial Token Mark integration
- 2026-01-21: Fixed data format (f0 keys, id_to_label)
