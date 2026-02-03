# Model Architecture - TranSTR với TokenMark (SoM)

## 📋 Tổng quan

Code đang tuning **TranSTR (Transformer-based VideoQA)** với **TokenMark (Set-of-Mark) injection** cho CausalVidQA dataset.

---

## 🏗️ Model Architecture

### **Base Model: VideoQAmodel (TranSTR)**

```
VideoQAmodel
├── Text Encoder: DeBERTa-base
├── Visual Encoders: ViT features (1024 dim)
├── Transformer Decoders (hierarchical)
├── Answer Classifier
└── TokenMark Injector (optional)
```

---

## 🔧 Components Chi Tiết

### 1. **Text Encoder**
- **Model**: `microsoft/deberta-base`
- **Output dim**: 768
- **Projection**: 768 → 768 (d_model)
- **Freeze**: `False` (trainable)
- **Pool mode**: 1 (mean pooling)

### 2. **Visual Features**
- **Frame features**: ViT features
  - Input: `[B, 16, 1024]` (16 frames, 1024 dim)
  - Resize: `1024 → 768` (d_model)
  - TopK selection: `16 → 5` frames
- **Object features**: Object detection features
  - Input: `[B, 16, 20, 2053]` (16 frames, 20 objects, 2053 dim)
  - Resize: `2053 → 768` (d_model)
  - TopK selection: `20 → 12` objects per frame

### 3. **Transformer Architecture**

#### **Hierarchical Decoders:**
```
1. Frame Decoder
   - Input: [B, 16, 768] frame features
   - Query: [B, seq_len, 768] question features
   - Output: [B, 16, 768] + attention weights
   - TopK: Select top 5 frames → [B, 5, 768]

2. Object Decoder
   - Input: [B*5, 20, 768] object features (flattened)
   - Query: [B*5, seq_len, 768] question (repeated)
   - Output: [B*5, 20, 768] + attention weights
   - TopK: Select top 12 objects → [B, 5, 12, 768]

3. Frame-Object Decoder (fo_decoder)
   - Input: [B, 5, 768] frame + [B, 5, 12, 768] objects
   - Output: [B, 5, 768] hierarchical features

4. VL Encoder (Vision-Language Fusion)
   - Input: [B, 5+seq_len, 768] (frames + question)
   - Output: [B, 5+seq_len, 768] fused memory

5. Answer Decoder
   - Input: [B, 5, 768] answer queries
   - Memory: [B, 5+seq_len, 768] from VL encoder
   - Output: [B, 5, 768] answer features
```

#### **Transformer Config:**
- **d_model**: 768
- **nheads**: 8
- **num_encoder_layers**: 2
- **num_decoder_layers**: 2
- **activation**: gelu
- **normalize_before**: True
- **dropout**: 0.3
- **encoder_dropout**: 0.3

### 4. **TopK Selection**
- **Frame TopK**: 5 frames (from 16)
  - Method: `PerturbedTopK` (differentiable)
  - Hard eval: `HardtopK` (non-differentiable)
- **Object TopK**: 12 objects (from 20)
  - Method: `PerturbedTopK` (differentiable)
  - Hard eval: `HardtopK` (non-differentiable)

### 5. **TokenMark (SoM) Injector** (Optional)
- **Enabled**: `use_som = True` (if SoM data available)
- **num_marks**: 16
- **Injection points**:
  - After frame resize & topK selection
  - After object resize & topK selection
- **Parameters**:
  - `gamma_frame`: 0.1 (learnable)
  - `gamma_obj`: 0.1 (learnable)
  - `palette`: 16 × 768 learnable embeddings

### 6. **Answer Classifier**
- **Input**: [B, 5, 768] answer features
- **Output**: [B, 5] logits (5 answer choices)
- **Layer**: `Linear(768, 1)` → squeeze

---

## 📊 Model Hyperparameters

### **Architecture:**
```python
d_model = 768
nheads = 8
num_encoder_layers = 2
num_decoder_layers = 2
activation = 'gelu'
normalize_before = True
dropout = 0.3
encoder_dropout = 0.3
```

### **Feature Dimensions:**
```python
frame_feat_dim = 1024  # ViT features
obj_feat_dim = 2053    # Object detection (2048 + 5 bbox)
word_dim = 768         # DeBERTa output
```

### **Selection:**
```python
topK_frame = 5   # Select 5 frames from 16
topK_obj = 12    # Select 12 objects from 20
frames = 16      # Total frames loaded
objs = 20        # Max objects per frame
```

### **Training:**
```python
batch_size = 8
learning_rate = 1e-5
weight_decay = 1e-4
epochs = 20
patience = 5
gamma = 0.1      # LR scheduler factor
```

---

## 🔄 Forward Pass Flow

```
1. Input Processing
   ├── Frame: [B, 16, 1024] → resize → [B, 16, 768]
   ├── Object: [B, 16, 20, 2053] → (keep for now)
   └── Question: text → DeBERTa → [B, seq_len, 768]

2. Frame Decoder + TopK
   ├── frame_local: [B, 16, 768]
   ├── frame_att: attention weights
   └── TopK selection → [B, 5, 768]

3. Object Processing
   ├── Select objects for top 5 frames → [B, 5, 20, 2053]
   └── Resize → [B, 5, 20, 768]

4. ⚠️ SoM Injection (if enabled)
   ├── Inject into frame_local: [B, 5, 768]
   └── Inject into obj_local: [B, 5, 20, 768]

5. Object Decoder + TopK
   ├── obj_local: [B*5, 20, 768]
   └── TopK selection → [B, 5, 12, 768]

6. Hierarchy Grouping
   ├── fo_decoder: [B, 5, 768] + [B, 5, 12, 768]
   └── Output: [B, 5, 768]

7. Vision-Language Fusion
   ├── Concatenate: [B, 5, 768] + [B, seq_len, 768]
   ├── VL encoder: [B, 5+seq_len, 768]
   └── Memory: [B, 5+seq_len, 768]

8. Answer Decoding
   ├── Answer queries: [B, 5, 768]
   ├── Answer decoder: [B, 5, 768]
   └── Classifier: [B, 5] logits
```

---

## 📈 Model Size

### **Parameters:**
- **Total**: ~110-120M parameters
- **Trainable**: ~110-120M (text encoder not frozen)
- **SoM injector**: ~16 × 768 = 12K additional parameters

### **Breakdown:**
- DeBERTa-base: ~86M
- Transformer layers: ~20M
- Feature resizers: ~2M
- SoM injector: ~12K
- Classifier: ~4K

---

## 🎯 Key Features

### **1. Hierarchical Attention**
- Frame-level attention → Object-level attention
- Multi-scale feature fusion

### **2. Differentiable TopK**
- `PerturbedTopK` for training (soft selection)
- `HardtopK` for evaluation (hard selection)

### **3. TokenMark Injection**
- Learnable entity embeddings
- Spatial mask-based injection
- Frame and object feature enhancement

### **4. Multi-modal Fusion**
- Vision-Language encoder
- Cross-attention between video and text
- Answer-specific decoding

---

## 🔍 Model Variants

### **Current Configuration:**
- **Text**: DeBERTa-base (trainable)
- **Visual**: ViT features (1024 dim)
- **SoM**: Enabled (if data available)
- **TopK**: 5 frames, 12 objects

### **Alternative Configurations:**
- **Text**: RoBERTa-base, BERT-base (configurable)
- **Visual**: ResNet + Motion (4096 dim) - not used in current setup
- **SoM**: Disabled (use_som=False)
- **TopK**: Configurable (topK_frame, topK_obj)

---

## 📝 Notes

1. **Model name**: TranSTR (Transformer-based VideoQA)
2. **Dataset**: CausalVidQA
3. **Task**: Multiple-choice VideoQA (5 choices)
4. **Evaluation**: Per-question-type accuracy (Description, Explanation, PAR, CAR, Acc_ALL)
5. **Special feature**: TokenMark (SoM) for explicit entity grounding

---

## 🔗 References

- TranSTR paper: [link to paper]
- DeBERTa: https://huggingface.co/microsoft/deberta-base
- TokenMark (SoM): Set-of-Mark prompting for visual grounding
- CausalVidQA: Causal Video Question Answering dataset
