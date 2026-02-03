# TokenMark (SoM) Issues - Low Accuracy Fix Guide

## 🔴 Các vấn đề phổ biến gây score thấp

### 1. **frame_feat_dim không khớp** ⚠️ CRITICAL

**Vấn đề:**
- Model mặc định expect `frame_feat_dim=4096` (ResNet + Motion concatenated)
- Nhưng ViT features chỉ có `1024` dimensions
- Nếu không set đúng, `frame_resize` layer sẽ có kích thước sai → model không học được

**Fix:**
```python
# Trong Config class (Cell 5):
frame_feat_dim = 1024  # ✅ Đúng cho ViT features
```

**Kiểm tra:**
- Chạy Cell 6.5 diagnostic
- Xem "Model frame_feat_dim" có = 1024 không

---

### 2. **SoM data không được load** ⚠️ HIGH

**Vấn đề:**
- Nếu `som_data` là `None` cho hầu hết samples, SoM injection không xảy ra
- Model vẫn chạy nhưng không có benefit từ Token Marks

**Fix:**
```python
# Kiểm tra trong Cell 6.5:
# - SoM available phải > 50% samples
# - Nếu < 50%, check đường dẫn SOM_FEATURE_PATH
```

**Debug:**
```python
# Thêm vào Cell 6 sau khi tạo datasets:
for i in range(10):
    sample = train_ds[i]
    som_data = sample[6]
    print(f"Sample {i}: som_data is {'NOT None' if som_data else 'None'}")
```

---

### 3. **use_som flag không nhất quán** ⚠️ HIGH

**Vấn đề:**
- `args.use_som` và `model.use_som` phải giống nhau
- Nếu khác nhau, training loop có thể không pass `som_data` vào model

**Fix:**
```python
# Trong Cell 7, sau khi tạo model:
assert args.use_som == model.use_som, "use_som flags must match!"

# Trong training loop (Cell 4):
if use_som and som_data is not None:  # ✅ Đúng
    out = model(ff, of, q, a, som_data=som_data)
else:
    out = model(ff, of, q, a)
```

---

### 4. **Gamma values quá nhỏ** ⚠️ MEDIUM

**Vấn đề:**
- `gamma_frame` và `gamma_obj` quá nhỏ → injection effect không đáng kể
- Default `gamma_init=0.1` có thể quá nhỏ

**Fix:**
```python
# Trong networks/som_injection.py, SoMInjector.__init__:
gamma_init=0.5  # Thử tăng từ 0.1 lên 0.5

# Hoặc sau khi tạo model:
if hasattr(model, 'som_injector'):
    with torch.no_grad():
        model.som_injector.gamma_frame.data.fill_(0.5)
        model.som_injector.gamma_obj.data.fill_(0.5)
```

**Kiểm tra:**
- Chạy Cell 6.5, xem "gamma_frame" và "gamma_obj" values
- Nếu < 0.01, injection gần như không có effect

---

### 5. **idx_frame shape không đúng** ⚠️ MEDIUM

**Vấn đề:**
- SoMInjector expect `idx_frame: [B, F_orig, frame_topK]`
- Nhưng sau `frame_sorter`, shape có thể khác

**Fix:**
```python
# Trong model.py forward(), sau frame topK selection:
# idx_frame shape should be [B, F, frame_topK] = [B, 16, 5]

# Verify trong SoMInjector.forward():
if idx_frame is not None:
    assert idx_frame.shape == (B, F_orig, frame_topK), \
        f"idx_frame shape {idx_frame.shape} != expected {(B, F_orig, frame_topK)}"
```

---

### 6. **Entity ID mapping sai** ⚠️ MEDIUM

**Vấn đề:**
- Entity IDs trong masks có thể non-contiguous (1, 3, 5)
- `entity_to_mark` mapping phải handle đúng

**Fix:**
- Code đã handle trong `get_active_mark_embeddings()`, nhưng verify:
```python
# Trong SoMInjector, check entity_to_mark:
entity_ids = sorted(entity_names.keys())  # [1, 3, 5]
entity_to_mark = {eid: idx for idx, eid in enumerate(entity_ids)}  # {1:0, 3:1, 5:2}
```

---

### 7. **Output không học được (std quá nhỏ)** ⚠️ CRITICAL

**Vấn đề:**
- Nếu output std < 0.1, model không học được gì
- Có thể do:
  - Learning rate quá nhỏ
  - Gradient bị vanish
  - Features không được normalize đúng

**Fix:**
```python
# Kiểm tra trong Cell 6.5:
# - Output std phải > 0.5
# - Nếu < 0.1, thử:
#   1. Tăng learning rate: lr = 5e-5
#   2. Giảm dropout: dropout = 0.1
#   3. Check gradient flow
```

---

## 🔧 Quick Fix Checklist

Chạy Cell 6.5 diagnostic và check:

- [ ] `frame_feat_dim == 1024` (cho ViT)
- [ ] `SoM available > 50%` samples
- [ ] `args.use_som == model.use_som`
- [ ] `gamma_frame > 0.01` và `gamma_obj > 0.01`
- [ ] `Output std > 0.5`
- [ ] Không có NaN/Inf trong output
- [ ] SoM data structure đúng (có `frame_masks` và `entity_names`)

---

## 🐛 Debug Steps

### Step 1: Verify SoM data loading
```python
# Thêm vào Cell 6:
sample = train_ds[0]
som_data = sample[6]
print(f"SoM data: {som_data}")
if som_data:
    print(f"  Keys: {som_data.keys()}")
    print(f"  Frame masks: {list(som_data['frame_masks'].keys())[:5]}")
    print(f"  Entity names: {som_data['entity_names']}")
```

### Step 2: Test forward pass với SoM
```python
# Thêm vào Cell 7 sau khi tạo model:
model.eval()
batch = next(iter(train_loader))
ff, of, q, a, ans_id, _, som_data = batch
ff, of = ff.to(device), of.to(device)

# Test với SoM
out_with_som = model(ff, of, q, a, som_data=som_data)
print(f"With SoM: {out_with_som.mean():.2f}, std: {out_with_som.std():.2f}")

# Test không SoM
out_no_som = model(ff, of, q, a, som_data=None)
print(f"No SoM: {out_no_som.mean():.2f}, std: {out_no_som.std():.2f}")

# So sánh
diff = (out_with_som - out_no_som).abs().mean()
print(f"Difference: {diff:.4f}")
# Nếu diff < 0.01, SoM injection không có effect!
```

### Step 3: Check gradient flow
```python
# Thêm vào training loop (Cell 10), sau loss.backward():
if ep == 1 and batch_idx == 0:
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm:.4f}")
    # Nếu < 0.1, gradient quá nhỏ → model không học
```

---

## 📊 Expected Results

### Baseline (không SoM):
- Acc_ALL: ~35-40% (tùy dataset)
- Description: ~40-50%
- Explanation: ~35-45%

### Với SoM (nếu hoạt động đúng):
- Acc_ALL: +2-5% improvement
- Description: +1-3%
- Explanation: +2-4%
- PAR/CAR: +3-6% (vì SoM giúp entity grounding)

### Nếu score < 20%:
- ❌ Model không học được → check learning rate, gradient flow
- ❌ Features sai → check frame_feat_dim
- ❌ Data loading sai → check DataLoader

---

## 🎯 Most Likely Issues (theo thứ tự)

1. **frame_feat_dim != 1024** → Model resize layer sai → không học được
2. **SoM data missing** → Injection không xảy ra → không có benefit
3. **use_som flags mismatch** → SoM không được pass vào model
4. **Output std quá nhỏ** → Model không học → check LR, dropout
5. **Gamma quá nhỏ** → Injection effect không đáng kể

---

## ✅ Final Checklist

Trước khi train lại:

- [ ] Chạy Cell 6.5 diagnostic
- [ ] Fix tất cả warnings
- [ ] Verify SoM data > 50% available
- [ ] Test forward pass với/không SoM
- [ ] Check output std > 0.5
- [ ] Verify gradient flow (grad norm > 0.1)
- [ ] Set learning rate phù hợp (1e-5 cho DeBERTa)
- [ ] Monitor training loss giảm dần

---

## 📝 Notes

- SoM injection chỉ giúp nếu:
  1. SoM data có sẵn cho > 50% samples
  2. Entity grounding thực sự quan trọng cho câu hỏi
  3. Model baseline đã hoạt động tốt (> 30% acc)

- Nếu baseline accuracy đã thấp (< 20%), fix baseline trước:
  - Check data loading
  - Check feature dimensions
  - Check learning rate
  - Check model architecture
