# NCOD + HUM Integration Planning — TranSTR

## 1. Mục tiêu

Tích hợp **HUM (High Uncertainty Module)** từ UCT vào pipeline **TranSTR + NCOD** hiện tại:

- **LUM (NCOD hiện tại)**: giảm trọng số mẫu nhiễu thông thường (Descriptive/Explanatory).
- **HUM (mới)**: tăng penalty cho mẫu khó/bất định cao (Predictive/Counterfactual + Reason).
- Không thay đổi kiến trúc TranSTR, không phá DataLoader, chỉ sửa hàm L1.

---

## 2. Kiểm tra code hiện tại (căn cứ thực tế)

### 2.1 DataLoader.py — `__getitem__`

Hiện tại DataLoader có **2 nhánh return** tuỳ `use_cached`:

```python
# Nhánh A — raw text (notebook hiện dùng):
return ff, of, qns, ans_word, ans_id, qns_key, idx
# 7 phần tử, qns_key dạng: "vid_descriptive", "vid_predictive_reason", ...

# Nhánh B — cached text (nếu dùng text_feature_path):
return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key, idx
# 9 phần tử
```

**Notebook hiện dùng Nhánh A (7 phần tử)**. Unpack trong train:

```python
ff, of, q, a, ans_id, _qns_key, sample_indices = batch
```

`_qns_key` đang bị bỏ qua (`_`). → **Để tạo `is_hard` mask, cần unpack `_qns_key` thành `qns_keys`**.

### 2.2 `qns_key` format thực tế

```
{video_id}_{type}
```

Giá trị `type` (từ annotation parsing loop trong DataLoader):

| type | Nhóm HUM | Mô tả |
|------|-----------|--------|
| `descriptive` | LUM (easy) | Câu mô tả |
| `explanatory` | LUM (easy) | Câu giải thích |
| `predictive` | **HUM (hard)** | Dự đoán đáp án |
| `predictive_reason` | **HUM (hard)** | Dự đoán lý do |
| `counterfactual` | **HUM (hard)** | Phản thực đáp án |
| `counterfactual_reason` | **HUM (hard)** | Phản thực lý do |

**Quan trọng**: cả `_reason` variant cũng thuộc HUM — phải check substring `predictive` hoặc `counterfactual` trong `qns_key`, không chỉ exact match.

### 2.3 `train_epoch_ncod` hiện tại — code thực

```python
def train_epoch_ncod(model, opt_model, opt_U, U, loader, device, epoch, accumulation_steps=4):
    ...
    for batch_idx, batch in enumerate(pbar):
        ff, of, q, a, ans_id, _qns_key, sample_indices = batch   # _qns_key bị bỏ qua
        ...
        logits = model(ff, of, q, a)              # [B, 5]
        probs  = F.softmax(logits, dim=1)         # [B, 5]
        u_batch  = U[sample_indices].unsqueeze(1) # [B, 1]
        y_onehot = F.one_hot(tgt, num_classes=NUM_CHOICES).float()

        # L1 (LUM-only hiện tại) — update model
        shifted_probs = probs + (u_batch.detach() * y_onehot)
        shifted_probs = torch.clamp(shifted_probs, min=1e-12, max=1.0)
        L1 = -torch.mean(torch.sum(y_onehot * torch.log(shifted_probs), dim=1))
        scaled_L1 = L1 / accumulation_steps
        scaled_L1.backward()

        # L2 — update U
        probs_det  = probs.detach()
        shifted_det = probs_det + (u_batch * y_onehot)
        L2 = F.mse_loss(shifted_det, y_onehot)
        scaled_L2 = L2 / accumulation_steps
        scaled_L2.backward()

        # Step sau mỗi accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_model.step(); opt_model.zero_grad()
            opt_U.step();     opt_U.zero_grad()
            with torch.no_grad(): U.clamp_(0.0, 0.99)
```

---

## 3. Kiến trúc tổng quan sau khi tích hợp HUM

```
                     DataLoader batch
     (ff, of, q, a, ans_id, qns_keys, sample_indices)
                          │
                ┌─────────▼─────────┐
                │   Forward TranSTR  │
                │   logits [B, 5]    │
                └─────────┬─────────┘
                          │
              probs = softmax(logits)
              u_batch = U[sample_indices]        [B, 1]
              y_onehot = one_hot(ans_id)         [B, 5]
                          │
              ┌───────────▼────────────┐
              │  is_hard mask          │
              │  từ qns_keys:          │
              │  'predictive' or       │
              │  'counterfactual' ∈ key│
              └───┬───────────────┬────┘
                  │               │
         is_hard=False      is_hard=True
              │                   │
    ┌─────────▼──────┐   ┌────────▼─────────┐
    │  LUM (NCOD)    │   │  HUM (UCT)        │
    │                │   │                   │
    │ shifted_probs  │   │ ce_loss_per_sample│
    │ = probs        │   │ = -∑ y*log(probs) │
    │ + u.detach()*y │   │                   │
    │                │   │ hum_loss          │
    │ lum_loss       │   │ = (1+u.detach())  │
    │ = -∑ y*log(s)  │   │   * ce_loss       │
    └────────┬───────┘   └────────┬──────────┘
             │                    │
             └────── torch.where(is_hard, hum_loss, lum_loss) ──┐
                                                                 │
                         final_L1 = mean(fused_loss) / accum_steps
                                        │
                               L2 (không đổi)
                        probs.detach + u*y → MSE(y_onehot)
                                        │
                     opt_model.step()  opt_U.step()
                              U.clamp_(0, 0.99)
```

---

## 4. Thay đổi code cụ thể

### 4.1 Sửa `train_epoch_ncod` trong notebook (Cell 4)

**Thay đổi duy nhất**: unpack `qns_keys` thay vì `_qns_key`, thêm `is_hard` mask, tách L1 thành LUM/HUM.

```python
def train_epoch_ncod(model, opt_model, opt_U, U, loader, device, epoch, accumulation_steps=4):
    model.train()
    total_l1, total_l1_lum, total_l1_hum = 0, 0, 0
    total_l2, correct, total = 0, 0, 0

    opt_model.zero_grad()
    opt_U.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, batch in enumerate(pbar):
        # ← SỬA: unpack qns_keys thay vì _qns_key
        ff, of, q, a, ans_id, qns_keys, sample_indices = batch
        ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)
        sample_indices = sample_indices.long()

        # ← MỚI: is_hard mask từ qns_key
        # qns_key format: "{video_id}_{type}"
        # type chứa 'predictive' hoặc 'counterfactual' → HUM
        is_hard = torch.tensor(
            ['predictive' in k or 'counterfactual' in k for k in qns_keys],
            dtype=torch.bool, device=device
        )  # [B]

        # Forward
        logits = model(ff, of, q, a)              # [B, 5]
        probs  = F.softmax(logits, dim=1)         # [B, 5]

        u_batch  = U[sample_indices].unsqueeze(1) # [B, 1]
        y_onehot = F.one_hot(tgt, num_classes=NUM_CHOICES).float()  # [B, 5]

        # --- L1: update model ---
        # Base CE per sample [B]
        ce_per_sample = -torch.sum(
            y_onehot * torch.log(torch.clamp(probs, min=1e-12)), dim=1
        )

        # LUM branch (descriptive/explanatory): NCOD shift
        shifted_probs = probs + (u_batch.detach() * y_onehot)
        shifted_probs = torch.clamp(shifted_probs, min=1e-12, max=1.0)
        lum_loss_per_sample = -torch.sum(y_onehot * torch.log(shifted_probs), dim=1)  # [B]

        # HUM branch (predictive/counterfactual): penalty scaling
        hum_loss_per_sample = (1.0 + u_batch.detach().squeeze(1)) * ce_per_sample    # [B]

        # Fuse theo is_hard mask
        fused_loss = torch.where(is_hard, hum_loss_per_sample, lum_loss_per_sample)  # [B]
        L1 = fused_loss.mean()

        # Log sub-losses (detached)
        n_hard = is_hard.sum().item()
        n_easy = (~is_hard).sum().item()
        l1_hum = hum_loss_per_sample[is_hard].mean().item() if n_hard > 0 else 0.0
        l1_lum = lum_loss_per_sample[~is_hard].mean().item() if n_easy > 0 else 0.0

        scaled_L1 = L1 / accumulation_steps
        scaled_L1.backward()

        # --- L2: update U (giữ nguyên) ---
        probs_det   = probs.detach()
        shifted_det = probs_det + (u_batch * y_onehot)
        L2 = F.mse_loss(shifted_det, y_onehot)
        scaled_L2 = L2 / accumulation_steps
        scaled_L2.backward()

        # Tracking
        total_l1     += L1.item()
        total_l1_hum += l1_hum
        total_l1_lum += l1_lum
        total_l2     += L2.item()
        correct += (logits.argmax(-1) == tgt).sum().item()
        total   += tgt.size(0)

        # Step optimizers
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_model.step(); opt_model.zero_grad()
            opt_U.step();     opt_U.zero_grad()
            with torch.no_grad(): U.clamp_(0.0, 0.99)

        pbar.set_postfix({
            'L1': total_l1 / (batch_idx + 1),
            'L2': total_l2 / (batch_idx + 1),
            'acc': correct / total * 100
        })

        if batch_idx % 50 == 0:
            wandb.log({
                'batch_L1': L1.item(),
                'batch_L1_LUM': l1_lum,
                'batch_L1_HUM': l1_hum,
                'batch_L2': L2.item(),
                'batch_acc': (logits.argmax(-1) == tgt).float().mean().item() * 100,
                'batch': epoch * len(loader) + batch_idx
            })

    n = len(loader)
    return total_l1/n, total_l1_lum/n, total_l1_hum/n, total_l2/n, correct/total*100
```

**Diff so với code hiện tại**:

| Dòng | Cũ | Mới |
|------|----|-----|
| Unpack | `ff, of, q, a, ans_id, _qns_key, sample_indices` | `ff, of, q, a, ans_id, qns_keys, sample_indices` |
| Sau forward | — | Tạo `is_hard` tensor từ `qns_keys` |
| L1 | 1 công thức (LUM) | 2 nhánh: LUM + HUM, fuse bằng `torch.where` |
| Return | `avg_l1, avg_l2, acc` | `avg_l1, avg_l1_lum, avg_l1_hum, avg_l2, acc` |

### 4.2 Cập nhật training loop (Cell 9) — unpack return mới

```python
# Cũ:
avg_l1, avg_l2, train_acc = train_epoch_ncod(...)
loss = avg_l1

# Mới:
avg_l1, avg_l1_lum, avg_l1_hum, avg_l2, train_acc = train_epoch_ncod(...)
loss = avg_l1

wandb.log({
    ...
    'train_L1': avg_l1,
    'train_L1_LUM': avg_l1_lum,   # loss của D+E
    'train_L1_HUM': avg_l1_hum,   # loss của P+C
    'train_L2': avg_l2,
    ...
})
print(f'L1: {avg_l1:.4f} (LUM={avg_l1_lum:.4f} | HUM={avg_l1_hum:.4f}) | L2: {avg_l2:.4f} | '
      f'Train: {train_acc:.1f}% | Val: {val_acc:.1f}%')
```

### 4.3 DataLoader.py — không cần sửa gì thêm

DataLoader đã trả `qns_key` ở vị trí thứ 6 (index -2). Format `{video_id}_{type}` với type đã bao gồm `predictive_reason`, `counterfactual_reason` → `is_hard` check substring là đúng.

### 4.4 Không sửa

| File | Lý do |
|------|-------|
| `networks/model.py` | Forward pass không đổi |
| `DataLoader.py` | Đã trả đủ `qns_key` và `idx` |
| `eval_epoch` | Không dùng U và không cần is_hard |

---

## 5. Hyperparameter HUM

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| `ncod_u_lr` | 0.1 | Giữ nguyên |
| `ncod_u_clamp_max` | 0.99 | Giữ nguyên |
| `hum_alpha` | 1.0 | Hệ số HUM: `(1 + alpha * u)`. Bắt đầu bằng 1, thử 2.0 nếu cần |

Thêm vào Cell 5 (Config):

```python
args.hum_alpha = 1.0  # scale factor cho HUM penalty
```

Và dùng trong HUM branch:

```python
hum_loss_per_sample = (1.0 + args.hum_alpha * u_batch.detach().squeeze(1)) * ce_per_sample
```

---

## 6. W&B Logging & Phân tích U theo qtype

### 6.1 Log W&B mỗi epoch (Cell 9)

```python
with torch.no_grad():
    u_np = U.detach().cpu().numpy()

    # Tách U theo qtype từ train_ds
    u_by_qtype = {}
    for i, row in enumerate(train_ds.sample_list.itertuples()):
        qt = row.type
        if qt not in u_by_qtype:
            u_by_qtype[qt] = []
        u_by_qtype[qt].append(u_np[i])

    u_log = {
        'U/mean': float(u_np.mean()),
        'U/max':  float(u_np.max()),
        'U/std':  float(u_np.std()),
        'U/histogram': wandb.Histogram(u_np),
    }
    for qt, vals in u_by_qtype.items():
        u_log[f'U/{qt}/mean'] = float(np.mean(vals))
    wandb.log(u_log)
```

### 6.2 Phân tích noisy sample (Cell 12 — sau train)

```python
with torch.no_grad():
    u_np = U.detach().cpu().numpy()
    df = train_ds.sample_list.copy().reset_index(drop=True)
    df['U_value'] = u_np

    # Phân phối U theo nhóm LUM vs HUM
    df['group'] = df['type'].apply(
        lambda t: 'HUM' if ('predictive' in t or 'counterfactual' in t) else 'LUM'
    )

    print("=== U stats by group ===")
    print(df.groupby('group')['U_value'].describe().round(6))
    print("\n=== U stats by qtype ===")
    print(df.groupby('type')['U_value'].describe().round(6))

    # Log to W&B
    table = wandb.Table(dataframe=df[['video_id','type','group','U_value']].head(500))
    wandb.log({'U_by_group_table': table})
```

---

## 7. Thứ tự thực hiện

| Step | File | Thay đổi |
|------|------|----------|
| 1 | Notebook Cell 4 | Unpack `qns_keys`, thêm `is_hard`, sửa L1 → LUM+HUM |
| 2 | Notebook Cell 9 | Unpack return 5 giá trị, log `L1_LUM`/`L1_HUM` |
| 3 | Notebook Cell 9 | Log U stats theo qtype |
| 4 | Notebook Cell 12 | Phân tích U theo group LUM/HUM |

**Không sửa**: `DataLoader.py`, `networks/*.py`, `eval_epoch`.

---

## 8. Checklist trước khi chạy

- [ ] `qns_keys` được unpack (không còn `_qns_key`)
- [ ] `is_hard` check `'predictive' in k or 'counterfactual' in k` — bao gồm cả `_reason`
- [ ] LUM branch: `u_batch.detach()` khi tính `shifted_probs`
- [ ] HUM branch: `u_batch.detach().squeeze(1)` trước khi nhân với `ce_per_sample` [B]
- [ ] `torch.where(is_hard, hum_loss_per_sample, lum_loss_per_sample)` — shape [B] nhất quán
- [ ] L2 không đổi: `probs.detach()`, MSE loss
- [ ] `U.clamp_(0.0, 0.99)` sau mỗi optimizer step
- [ ] Return 5 giá trị từ `train_epoch_ncod`
- [ ] Training loop unpack 5 giá trị
- [ ] W&B log thêm `train_L1_LUM`, `train_L1_HUM`

---

## 9. Rủi ro và cách xử lý

| Rủi ro | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| Shape mismatch trong `torch.where` | `is_hard [B]`, `lum/hum [B]` phải cùng shape 1D | Đảm bảo `squeeze(1)` trước khi nhân |
| `l1_hum = 0.0` mọi epoch | Batch không có mẫu hard | Tăng batch size hoặc check DataLoader shuffle |
| HUM loss quá lớn → gradient explosion | `U` cao + `alpha` lớn | Giảm `hum_alpha`, kiểm tra `grad_norm` |
| `train_acc` tụt sau khi thêm HUM | HUM penalize quá mạnh | Thử `hum_alpha = 0.5` |
| NaN loss | `ce_per_sample` âm / `probs = 0` | Đảm bảo `clamp(probs, min=1e-12)` trước log |

---

## 10. Tóm tắt diff tối thiểu

```diff
# train_epoch_ncod — Cell 4

- ff, of, q, a, ans_id, _qns_key, sample_indices = batch
+ ff, of, q, a, ans_id, qns_keys, sample_indices = batch

+ is_hard = torch.tensor(
+     ['predictive' in k or 'counterfactual' in k for k in qns_keys],
+     dtype=torch.bool, device=device
+ )

+ ce_per_sample = -torch.sum(y_onehot * torch.log(torch.clamp(probs, 1e-12)), dim=1)
  shifted_probs = probs + (u_batch.detach() * y_onehot)
  shifted_probs = torch.clamp(shifted_probs, min=1e-12, max=1.0)
- L1 = -torch.mean(torch.sum(y_onehot * torch.log(shifted_probs), dim=1))
+ lum_loss_per_sample = -torch.sum(y_onehot * torch.log(shifted_probs), dim=1)
+ hum_loss_per_sample = (1.0 + u_batch.detach().squeeze(1)) * ce_per_sample
+ L1 = torch.where(is_hard, hum_loss_per_sample, lum_loss_per_sample).mean()

- return avg_l1, avg_l2, acc
+ return avg_l1, avg_l1_lum, avg_l1_hum, avg_l2, acc
```
