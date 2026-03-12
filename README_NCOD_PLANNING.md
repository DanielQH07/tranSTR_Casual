# NCOD Integration Planning — TranSTR + Noise Detection

## 1. Mục tiêu

Tích hợp cơ chế **chiết khấu U (discount parameter)** từ bài báo NCOD vào pipeline TranSTR hiện tại để:
- Phát hiện các mẫu bị gán nhãn sai (noisy labels) trong CausalVidQA.
- Theo dõi phân phối U qua W&B để xác định tỷ lệ nhiễu thực tế.
- Giữ nguyên pipeline TranSTR gốc, chỉ thay đổi hàm loss và thêm tham số U.

**Không dùng**: Soft label / centroid similarity của NCOD (vì CausalVidQA là multiple-choice, không có centroid cụm rõ ràng).

---

## 2. Tổng quan thiết kế

```
                        ┌─────────────────────────┐
                        │    DataLoader (sửa)      │
                        │  trả thêm sample_index   │
                        └────────────┬────────────┘
                                     │
               ┌─────────────────────▼─────────────────────┐
               │              Forward Pass                  │
               │  TranSTR gốc → logits [B, 5]              │
               └─────────────────────┬─────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                  │
            ┌───────▼───────┐                  ┌──────▼───────┐
            │   L1 Loss     │                  │   L2 Loss    │
            │ (update model)│                  │ (update U)   │
            │               │                  │              │
            │ shifted_probs │                  │ MSE target   │
            │ = softmax(logits)                │ = one_hot    │
            │   + U.detach()│                  │ shifted =    │
            │   * y_onehot  │                  │ softmax.detach│
            │               │                  │   + U*y      │
            │ CE(shifted, y)│                  │ MSE(shifted,y)│
            └───────┬───────┘                  └──────┬───────┘
                    │                                  │
            opt_TranSTR.step()                  opt_U.step()
                                                U.clamp_(0, 0.99)
```

---

## 3. Các file cần sửa / tạo mới

### 3.1 Sửa: `DataLoader.py`

**Mục đích**: Trả thêm `sample_index` (vị trí trong dataset) để mapping đúng U[i].

**Vị trí sửa**: method `__getitem__`

**Thay đổi cụ thể**:
```python
# TRƯỚC (return hiện tại):
return ff, of, qns, ans_word, ans_id, qns_key

# SAU (thêm idx):
return ff, of, qns, ans_word, ans_id, qns_key, idx
```

**Lưu ý**: `idx` chính là tham số `idx` của `__getitem__(self, idx)`, không cần tạo thêm gì.

---

### 3.2 Sửa: Notebook `train-transtr-different-feature.ipynb`

#### Cell 4 — Train/Eval functions

Thay đổi lớn nhất. Cần sửa:

1. **Unpack batch thêm `sample_indices`**:
```python
# TRƯỚC:
ff, of, q, a, ans_id, _ = batch

# SAU:
ff, of, q, a, ans_id, _qns_key, sample_indices = batch
```

2. **Thêm L1 loss (cập nhật model)**:
```python
logits = model(ff, of, q, a)
probs = F.softmax(logits, dim=1)

u_batch = U[sample_indices].unsqueeze(1)  # [B, 1]
y_onehot = F.one_hot(tgt, num_classes=5).float()  # [B, 5]

shifted_probs = probs + (u_batch.detach() * y_onehot)
shifted_probs = torch.clamp(shifted_probs, min=1e-12, max=1.0)
L1 = -torch.mean(torch.sum(y_onehot * torch.log(shifted_probs), dim=1))

opt_TranSTR.zero_grad()
L1.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt_TranSTR.step()
```

3. **Thêm L2 loss (cập nhật U)**:
```python
probs_det = probs.detach()
shifted_det = probs_det + (u_batch * y_onehot)
L2 = F.mse_loss(shifted_det, y_onehot)

opt_U.zero_grad()
L2.backward()
opt_U.step()

with torch.no_grad():
    U.clamp_(0.0, 0.99)
```

4. **Eval function giữ nguyên** (không dùng U khi eval).

#### Cell 5 — Config

Thêm các hyperparameter NCOD:
```python
# NCOD Config
ncod_u_lr = 0.1          # Learning rate cho U (SGD)
ncod_u_mean = 1e-8       # Khởi tạo U
ncod_u_std = 1e-9
ncod_u_clamp_max = 0.99  # Giới hạn trên của U
```

#### Cell 7 — Model + Optimizer

Thêm khởi tạo U và optimizer U:
```python
# Khởi tạo U
num_train_samples = len(train_ds)
U = torch.nn.Parameter(
    torch.abs(torch.randn(num_train_samples) * args.ncod_u_std + args.ncod_u_mean)
).to(device)

# 2 Optimizer riêng biệt
opt_TranSTR = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
     "lr": args.text_encoder_lr}
], lr=args.lr, weight_decay=args.decay)

opt_U = torch.optim.SGD([U], lr=args.ncod_u_lr)
```

#### Cell 8 — W&B Init

Thêm NCOD config vào wandb_config:
```python
wandb_config['ncod_enabled'] = True
wandb_config['ncod_u_lr'] = args.ncod_u_lr
wandb_config['ncod_u_clamp_max'] = args.ncod_u_clamp_max
```

#### Cell 9 — Training Loop

Cuối mỗi epoch thêm log U:
```python
with torch.no_grad():
    u_np = U.detach().cpu().numpy()
    wandb.log({
        'epoch': ep,
        # ... metrics hiện tại ...
        'U/mean': float(u_np.mean()),
        'U/max': float(u_np.max()),
        'U/min': float(u_np.min()),
        'U/std': float(u_np.std()),
        'U/pct_above_0.1': float((u_np > 0.1).mean() * 100),
        'U/pct_above_0.3': float((u_np > 0.3).mean() * 100),
        'U/pct_above_0.5': float((u_np > 0.5).mean() * 100),
        'U/histogram': wandb.Histogram(u_np),
    })
```

#### Cell 10 — Evaluation

Giữ nguyên `evaluate_detailed_v2`. Không dùng U khi eval.

#### Thêm Cell mới (Cell 12) — Noisy Sample Analysis

Cell mới sau Cell 11, phân tích top noisy samples:
```python
# Phân tích top câu hỏi bị nghi nhiễu
with torch.no_grad():
    u_np = U.detach().cpu().numpy()
    top_k = 100
    noisy_indices = u_np.argsort()[-top_k:][::-1]

    print(f"Top {top_k} samples with highest U (suspected noisy):")
    print("="*80)
    for rank, idx in enumerate(noisy_indices[:20]):
        row = train_ds.sample_list.iloc[idx]
        print(f"#{rank+1} | U={u_np[idx]:.4f} | vid={row['video_id']} | "
              f"type={row['type']} | Q={row['question'][:60]}...")

    # Save full list
    noisy_df = train_ds.sample_list.iloc[noisy_indices].copy()
    noisy_df['u_value'] = u_np[noisy_indices]
    noisy_df.to_csv('suspected_noisy_samples.csv', index=False)
    print(f"\nSaved suspected_noisy_samples.csv ({len(noisy_df)} rows)")

    # Log to W&B
    wandb.log({'noisy_samples_table': wandb.Table(dataframe=noisy_df.head(100))})
```

---

### 3.3 Không sửa (giữ nguyên)

| File | Lý do |
|------|-------|
| `networks/model.py` | Forward pass không đổi, U chỉ tác động ở loss |
| `networks/multimodal_transformer.py` | Không liên quan |
| `networks/attention.py` | Không liên quan |
| `networks/topk.py` | Không liên quan |
| `eval_mc.py` | Evaluation không dùng U |

---

## 4. Thứ tự sửa code (step-by-step)

### Step 1: Sửa DataLoader trả thêm index
- File: `DataLoader.py`
- Sửa: `__getitem__` return thêm `idx`
- Test: load 1 batch, verify có `sample_indices` tensor

### Step 2: Thêm NCOD config vào notebook
- Cell 5: thêm `ncod_u_lr`, `ncod_u_mean`, `ncod_u_std`, `ncod_u_clamp_max`

### Step 3: Khởi tạo U và opt_U
- Cell 7: tạo `U` parameter và `opt_U` (SGD)
- Tách `optimizer` thành `opt_TranSTR`

### Step 4: Sửa training loop
- Cell 4 (hoặc Cell 9): unpack batch mới, bi-level loss (L1 → model, L2 → U)
- Thêm grad clip cho model
- Thêm U.clamp_ sau mỗi step

### Step 5: Thêm W&B logging cho U
- Cell 8: thêm ncod config
- Cell 9: log U stats + histogram mỗi epoch

### Step 6: Thêm cell phân tích noisy samples
- Cell 12: in top-100 U, lưu CSV, log W&B table

### Step 7: Test chạy 1-2 epoch
- Verify U histogram xuất hiện trên W&B
- Verify model vẫn converge (train loss giảm)
- Verify U ban đầu gần 0, sau vài epoch bắt đầu phân tách

---

## 5. Cách đọc kết quả trên W&B

### 5.1 U Histogram (Bimodal test)

| Trạng thái | Ý nghĩa |
|-------------|----------|
| Tất cả U ≈ 0 (unimodal) | U chưa học được gì, cần chạy thêm epoch |
| 2 đỉnh: đỉnh lớn ≈ 0, đỉnh nhỏ ≈ 0.3-0.9 | **Thành công** — nhóm nhỏ là suspected noisy |
| Tất cả U tăng đều | Loss setup có vấn đề, kiểm tra lại L2 |

### 5.2 Val Accuracy so sánh

| Run | Dấu hiệu |
|-----|-----------|
| TranSTR gốc (CE) | Val acc tăng → đạt đỉnh → giảm (overfit noise) |
| TranSTR + NCOD | Val acc tăng → ổn định hoặc tăng chậm (noise bị cách ly) |

### 5.3 Top noisy samples

- In ra video + question của top-100 U cao nhất.
- Mở video xem bằng mắt → đa số sẽ thấy đáp án gốc thực sự sai hoặc mơ hồ.

---

## 6. Hyperparameter khuyến nghị

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| `ncod_u_lr` | 0.1 | SGD, theo phụ lục bài báo |
| `ncod_u_mean` | 1e-8 | Khởi tạo rất nhỏ |
| `ncod_u_std` | 1e-9 | Khởi tạo rất nhỏ |
| `ncod_u_clamp_max` | 0.99 | Theo Theorem 5.1 |
| `model_lr` | 1e-5 | Giữ nguyên như TranSTR gốc |
| `epochs` | 20-30 | Cần đủ epoch để U phân tách |

---

## 7. Checklist trước khi chạy

- [ ] DataLoader trả thêm `idx` → batch có `sample_indices`
- [ ] U được khởi tạo đúng size = `len(train_ds)`
- [ ] opt_TranSTR và opt_U tách riêng
- [ ] L1 loss: U.detach() khi tính shifted_probs
- [ ] L2 loss: probs.detach() khi tính shifted cho U
- [ ] U.clamp_(0, 0.99) sau mỗi optimizer step
- [ ] W&B log U histogram + mean/max/min mỗi epoch
- [ ] Eval không dùng U (chỉ CE hoặc argmax thuần)
- [ ] Cell phân tích noisy samples lưu CSV + log W&B table

---

## 8. Rủi ro và cách xử lý

| Rủi ro | Giải pháp |
|--------|-----------|
| U tăng quá nhanh → model không học | Giảm `ncod_u_lr` (thử 0.01) |
| U không tăng sau 10 epoch | Tăng `ncod_u_lr` hoặc kiểm tra L2 gradient |
| Train loss NaN | Kiểm tra `shifted_probs` clamp, eps đủ lớn |
| Batch unpack sai | Assert shape ngay đầu loop |
| Memory tăng do U | U chỉ là 1D tensor, không đáng kể |

---

## 9. Tóm tắt diff

```
DataLoader.py
  └── __getitem__: return thêm idx

Notebook (train-transtr-different-feature.ipynb):
  ├── Cell 4:  train_epoch → bi-level loss (L1 + L2)
  ├── Cell 5:  thêm NCOD hyperparams
  ├── Cell 7:  khởi tạo U, opt_U, tách opt_TranSTR
  ├── Cell 8:  wandb_config thêm NCOD fields
  ├── Cell 9:  log U stats mỗi epoch
  └── Cell 12: (MỚI) phân tích noisy samples

Không sửa:
  ├── networks/model.py
  ├── networks/*.py
  └── eval_mc.py
```
