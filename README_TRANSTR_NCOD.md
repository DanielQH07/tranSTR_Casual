# TranSTR + DINOv3 + NCOD (GA) - README (current)

Mô tả chi tiết pipeline trong notebook `train-transtr-different-feature_ncod_ga.ipynb` (branch `ncod`) hiện tại, bao gồm:
- Kiến trúc model đang dùng (TranSTR / `VideoQAmodel`)
- Cách DataLoader xuất dữ liệu
- Cơ chế NCOD (bi-level optimization với tham số U_i theo từng mẫu)
- Gradient Accumulation (GA)
- Optimizer/scheduler/checkpoint/resume
- Evaluation và noisy-sample analysis

---

## 1) Input data & feature (đầu vào)

Notebook sử dụng 2 loại feature thị giác:

### 1.1 Frame feature: DINOv3 .pt (1024D)
- Backbone frame: `openai/dinov3` với feature dim `FEAT_DIM = 1024`
- File lưu:
  - `CLIP_FEATURE_PATH = CLIP_MERGED_PATH` (ví dụ trong notebook: `/kaggle/working/dinov3_T16_dim1024_merge`)
  - DataLoader giả định có file dạng: `video_id.pt`
- Quan hệ shape (theo cách DataLoader sampling/padding):
  - Mỗi mẫu: `ff` có shape `[T_loaded, 1024]` với `T_loaded = topK_frame` (trong config: `topK_frame = 16`)
  - Nếu `nf > topK_frame`: lấy mẫu đều `np.linspace(0, nf-1, topK_frame)`
  - Nếu `nf < topK_frame`: pad bằng zeros tới `topK_frame`

### 1.2 Object feature: precomputed object detection features
- `object_feature_path = /kaggle/input/object-detection-causal-full`
- DataLoader quét/scan thư mục object features và tải theo `vid`:
  - Có thể là format “kaggle_subdirs” (mỗi subdir chứa `.pkl`)
  - Trả về object tensor `of` dạng:
    - `of`: `[topK_frame, obj_num, obj_feat_dim]`
    - `obj_num = objs = 20`
    - `obj_feat_dim = 2053` (2048 feature + 5 bbox dims sau `transform_bb`)
- Sampling/padding cho time dimension:
  - Duyệt frames theo `indices = linspace(0, num_frames-1, topK_frame)` hoặc `range(num_frames)` nếu ít frame
  - Pad tới `topK_frame` và pad tới `obj_num` (nếu thiếu objects)

### 1.3 Text/annotation: `text-annotation/QA`
- Đường dẫn: `ANNOTATION_PATH = /kaggle/input/text-annotation/QA`
- `VideoQADataset` đọc:
  - `text.json`: question + answer candidates theo các type
  - `answer.json`: ground-truth answer index
- Với “reason” (predictive_reason / counterfactual_reason):
  - DataLoader biến question thành `"Why?"`
  - Answer candidates lấy từ field `reason`

---

## 2) DataLoader output (DataLoader.py)

Trong notebook NCOD, dataset được tạo bằng:
- `VideoQADataset(split=..., n_query=5, obj_num=objs, sample_list_path=ANNOTATION_PATH, video_feature_path=..., object_feature_path=..., split_dir=..., topK_frame=16)`

Mỗi sample `__getitem__` trả về:

`(ff, of, qns, ans_word, ans_id, qns_key, idx)`

Trong đó:
- `ff`: `FloatTensor[topK_frame, 1024]` (ví dụ `[16, 1024]`)
- `of`: `FloatTensor[topK_frame, obj_num, 2053]` (ví dụ `[16, 20, 2053]`)
- `qns`: string question (dùng tokenizer trong model)
- `ans_word`: list 5 string answer options
- `ans_id`: int ground-truth index trong [0..4]
- `qns_key`: key dạng `f"{vid}_{type}"` (dùng cho logging/analysis)
- `idx`: index của sample trong dataset (dùng để mapping U_i)

### 2.1 Output `idx` cho NCOD U mapping
NCOD cần per-sample U_i. Notebook dùng:
- Batch trả `sample_indices` (stack từ `idx`)
- Khi resume, U được remap theo “stable keys”:
  - `train_sample_keys = [ f"{video_id}_{type}" ... ]`

---

## 3) Model kiến trúc: TranSTR (`networks/model.py::VideoQAmodel`)

Trong notebook, model tạo bằng:
- `model = VideoQAmodel(**cfg)` với `d_model = 768`
- Text encoder: `microsoft/deberta-base` (không freeze)
- `use_som` mặc định = False trong notebook NCOD này (không bật TokenMark)

### 3.1 Feature resize
- `frame_resize`: `FeatureResizer(frame_feat_dim -> d_model)`
  - `frame_feat_dim = 1024`
  - Output: `frame_feat` `[B, F_loaded, 768]`
- `obj_resize`: `FeatureResizer(obj_feat_dim -> d_model)`
  - `obj_feat_dim = 2053`
  - Input: `obj_feat` `[B, F_loaded, O, 2053]` -> Output `[B, F_loaded, O, 768]`

### 3.2 TopK selection (differentiable trong training)
Forward của `VideoQAmodel` (tóm tắt):
1) Frame decoder:
   - `frame_local, frame_att = frame_decoder(frame_feat, q_local, output_attentions=True)`
2) Chọn topK frame:
   - `frame_topK = select_frames = 5` (được set trong notebook qua `cfg['topK_frame']`)
3) Chọn topK object:
   - `obj_topK = topK_obj = 12`

### 3.3 Hierarchical fusion & answer decoding
- `fo_decoder` nhóm frame + object level
- `vl_encoder` fusion với question tokens
- `ans_decoder` decode theo answer queries
- `classifier`: output logits `[B, 5]`

---

## 4) NCOD training (bi-level optimization với U)

### 4.1 Tham số U
- Tạo: `U` có shape `[N_train]` (N_train = `len(train_ds)`)
- Init:
  - `U = abs(randn(N_train) * ncod_u_std + ncod_u_mean)`
- Clamp:
  - `U.clamp_(0.0, ncod_u_clamp_max)` sau mỗi step (ncod_u_clamp_max = 0.99)

### 4.2 Loss L1 và L2 trong `train_epoch_ncod`
Với batch:
- `logits = model(ff, of, q, a)` -> `[B, 5]`
- `probs = softmax(logits, dim=1)`
- `tgt = ans_id` (0..4)
- `u_batch = U[sample_indices]` -> `[B, 1]`
- `y_onehot = one_hot(tgt, num_classes=5).float()` -> `[B, 5]`

**L1 (update model params, U detach)**
- `shifted_probs = probs + (u_batch.detach() * y_onehot)`
- `shifted_probs = clamp(shifted_probs, 1e-12, 1.0)`
- `L1 = -mean( sum( y_onehot * log(shifted_probs) ) )`

**L2 (update U params, model probs detach)**
- `probs_det = probs.detach()`
- `shifted_det = probs_det + (u_batch * y_onehot)`
- `L2 = mse_loss(shifted_det, y_onehot)`

### 4.3 Optimizer 2 tầng (dual optimizers)
- Backward:
  - L1 backward -> chỉ update model (vì U detached)
  - L2 backward -> chỉ update U (vì probs_det detached)
- Step:
  - `opt_model.step()`
  - `opt_U.step()`

---

## 3.4) Kiến trúc tổng quan TranSTR + NCOD (GA)

```text
                              (Train dataloader, batch)
   (ff, of, qns, ans_word, ans_id, qns_key, idx)
            │
            │ 1) Forward TranSTR
            ▼
   VideoQAmodel(ff, of, qns, ans_word) → logits [B, 5]
            │
            │ 2) NCOD loss (bi-level)
            │
            │   probs = softmax(logits)
            │   y_onehot = one_hot(ans_id)         [B, 5]
            │   u_batch = U[idx]                  [B, 1]
            │
            ├─────────────────────────────────────────────┐
            │                                             │
            │   L1 (update MODEL, U detached)              │
            │   shifted_probs = probs + u_batch.detach * y
            │   L1 = -mean( y * log(shifted_probs) )       │
            │                                             │
            │   L2 (update U, model probs detached)        │
            │   shifted_det = probs.detach + u_batch * y
            │   L2 = MSE(shifted_det, y)                  │
            │                                             │
            └─────────────────────────────────────────────┘
                           │                   │
                           │ opt_model.step   │ opt_U.step
                           ▼                   ▼
                     (Transformer + DeBERTa)     U_i
                           │                   │
                           └────── U.clamp_(0, 0.99)

                     Scheduler: ReduceLROnPlateau(opt_model, monitor=val_acc)
                     Evaluation: eval_epoch ignores U (chỉ dùng model)
```

## 10) Tại sao train/val đứng quanh ~60% từ epoch ~7?

Mình không nhìn được curve của bạn trực tiếp trong README, nhưng theo đúng logic trong `train-transtr-different-feature_ncod_ga.ipynb` thì có vài nguyên nhân phổ biến khiến acc “đứng xoay vòng” sau một vài epoch:

1. **NCOD có thể “converge nhanh”**: `U` được khởi tạo rất nhỏ (`mean=1e-8`, `std=1e-9`) và bị clamp sau mỗi step (`U.clamp_(0.0, 0.99)`). Nếu sau epoch ~7 thì `L2` của U giảm rất thấp và thống kê `U/mean, U/std, U/max` gần như ổn định, thì về sau training gần như quay lại hành vi giống baseline (CE-like) và acc khó tăng thêm.

2. **LR/scheduler giảm không đúng nhịp cải thiện**: bạn dùng `ReduceLROnPlateau(opt_model, 'max', patience=5, factor=0.1)`. Nếu val_acc không cải thiện trong một khoảng đủ dài, LR sẽ giảm; đôi khi giảm sớm/quá muộn khiến model rơi vào vùng “plateau” và không thoát được.

3. **Effective objective đã bão hoà**: TranSTR + topK selection + decoder hierarchy thường học tốt các mẫu “easy” sớm. Khi còn lại chủ yếu là “hard/noisy”, val acc có thể dao động quanh một mức do nhiễu nhãn/thiếu thông tin thị giác.

4. **GA/TopK làm gradient có độ nhiễu**: TopK chọn frame/object ảnh hưởng mạnh luồng attention. Dù `PerturbedTopK` là differentiable, nhưng tín hiệu vẫn có độ biến thiên, nên train/val có thể dao động mà không tăng đều.

## 11) Mỗi epoch hiện tại đang train cái gì?

Trong mỗi epoch, ở hàm `train_epoch_ncod`:

1. **Đang train (cập nhật qua `opt_model`)**
   - Toàn bộ tham số của `VideoQAmodel`, gồm cả **DeBERTa text encoder** vì `freeze_text_encoder=False`.
   - Kể cả các module: `frame_resize`, `obj_resize`, `frame_decoder`, `obj_decoder`, `fo_decoder`, `vl_encoder`, `ans_decoder`, `classifier`, và các projection.

2. **Đang train (cập nhật qua `opt_U`)**
   - Tham số riêng `U` với shape `[len(train_ds)]` (mỗi sample có 1 giá trị U_i).
   - Sau mỗi step: `U.clamp_(0.0, 0.99)`.

3. **Scheduler**
   - Cuối mỗi epoch gọi `scheduler.step(val_acc)` để giảm LR của `opt_model` nếu val_acc plateau.

4. **Evaluation**
   - `eval_epoch` chỉ gọi `model(...)` và **không dùng U**.

## 12) DeBERTa có phải “tốt nhất” cho training hiện tại không?

- **DeBERTa-base là một lựa chọn khá mạnh và phù hợp** cho bài toán QA/question answering vì khả năng mô hình hóa ngữ nghĩa tốt.
- Tuy nhiên **“tốt nhất” không chắc đúng tuyệt đối**: plateau quanh ~60% có thể đến từ visual branch/object features, NCOD dynamics, hoặc objective design hơn là do text encoder.

Khuyến nghị kiểm chứng (ablation) nhanh:

1. So sánh `freeze_text_encoder=True` trong vài epoch đầu (hoặc giảm `text_encoder_lr` thấp hơn `lr`) để xem liệu text encoder đang “làm nhiễu” hay không.
2. Thử đổi backbone text encoder sang một biến thể mạnh khác (ví dụ `roberta-base` hoặc DeBERTa-v3-base nếu bạn có sẵn model).
3. Theo dõi `L1/L2` và thống kê `U` trên W&B: nếu `U` ổn định sớm thì bottleneck không nằm ở text encoder.

## 5) Gradient Accumulation (GA)

Notebook dùng GA để tăng effective batch size:
- micro-batch: `bs = 8`
- `accumulation_steps = 4`
- effective batch size = `32`

Implementation:
- Scale: `scaled_L1 = L1 / accumulation_steps`, `scaled_L2 = L2 / accumulation_steps`
- Backward mỗi micro-batch
- Step model và U mỗi `accumulation_steps`

---

## 6) Hyperparameters đang dùng (theo notebook)

### 6.1 TranSTR base
- `d_model = 768`
- `n_query = 5`
- Transformer:
  - `nheads = 8`
  - `num_encoder_layers = 2`
  - `num_decoder_layers = 2`
  - `dropout = 0.3`
  - `encoder_dropout = 0.3`
  - `activation = gelu`
  - `normalize_before = True`
- Sampling/selection:
  - `topK_frame (load) = 16`
  - `select_frames (model) = 5`
  - `obj_num = objs = 20`
  - `topK_obj = 12`

### 6.2 Training
- `epoch = 20`
- `lr = 1e-5` (AdamW)
- `decay = 1e-4` (weight decay)
- `patience = 5`
- `gamma = 0.1` (ReduceLROnPlateau factor)
- `accumulation_steps = 4`
- `num_workers = 4`

### 6.3 NCOD
- `ncod_u_lr = 0.1` (SGD cho U)
- `ncod_u_mean = 1e-8`
- `ncod_u_std = 1e-9`
- `ncod_u_clamp_max = 0.99`

---

## 7) Checkpoint & Resume

Notebook lưu:
- `latest_checkpoint_dinov3_ncod.ckpt` (mỗi epoch)
- `best_model_dinov3_ncod.ckpt` (theo val_acc)

Checkpoint lưu cả:
- `model_state_dict`
- `U` tensor
- `opt_model_state_dict`
- `opt_U_state_dict`
- `scheduler_state_dict`
- `epoch`, `best_acc`, `val_acc`, `train_acc`, `history`
- `train_sample_keys` để remap U khi resume

Resume:
- Nếu dataset sample order/size thay đổi -> remap U_old sang U_new dựa trên `"{video_id}_{type}"`.

---

## 8) Evaluation & Noisy-sample analysis

### 8.1 Evaluation
- `eval_epoch` không dùng U (chỉ forward model thường).
- Notebook tải best checkpoint trước evaluation.
- Metrics theo qtype (Description/Explanation/PAR/CAR/Acc_ALL).

### 8.2 Noisy-sample analysis (CELL 12)
- Lấy U_i và build bảng:
  - `sample_idx`, `qns_key`, `U_value`
- Log W&B Table:
  - `ncod_video_id_U_table`
- Save CSV:
  - `ncod_all_video_U.csv`
  - `ncod_noisy_samples.csv` (top-k theo U)
- Plot histogram:
  - `ncod_u_distribution.png`

---

## 9) Gói gọn: TranSTR-NCOD-GA đang làm gì

- Frame backbone: DINOv3 features (1024D)
- Object branch: precomputed object detection features (2053D)
- Model: TranSTR hierarchical decoders + answer decoding
- NCOD: học U_i cho từng sample, cập nhật bi-level (L1 model / L2 U) + clamp
- GA: micro-batch + accumulation_steps=4
- Inference/eval: bỏ qua U (chỉ dùng model đã được NCOD train)

