# Phân Tích Kiến Trúc: TranSTR gốc vs TranSTR-DN (Hiện tại)

Tài liệu này phân tích chi tiết các thay đổi về mặt kiến trúc, thiết kế và luồng dữ liệu (data flow) giữa mô hình **TranSTR gốc** (được giới thiệu trong bài báo *Invariant Grounding for Video Question Answering, CVPR 2022*) và mô hình **TranSTR-DN hiện tại** (đã được tích hợp DINOv3 và cơ chế NCOD).

---

## 1. Tóm Tắt Các Điểm Khác Biệt Cốt Lõi

| Thành Phần Kiến Trúc | TranSTR (Gốc) | TranSTR-DN (Hiện Tại) | Ý Nghĩa / Lợi Ích Của Sự Thay Đổi |
| :--- | :--- | :--- | :--- |
| **Visual Backbone (Frame)** | ResNet-101 (Appearance) + 3D-ResNet (Motion) | **DINOv3 ViT-L/14** | Chuyển từ supervised (nhắm vào object/motion cơ bản) sang self-supervised (hiểu semantics toàn cục sâu sắc hơn). DINOv3 sinh ra attention maps khu vực rất tốt cho việc chọn frame/object mấu chốt. |
| **Thứ Nguyên Feature Hình Ảnh (Frame Dimension)** | `4096` (Concat `2048` + `2048`) | **`1024`** | Giảm thiểu khối lượng tính toán, giúp `FeatureResizer` học map các semantic features vào `768` (d_model) dễ dàng hơn mà không bị nhiễu bởi các chiều không tương quan. |
| **Batch Size & VRAM Optimization** | `bs=16` hoặc `bs=32` thực sự | **Gradient Accumulation** (`bs=8`, `accum=4` -> `Effective BS=32`) | Cho phép huấn luyện với Batch Size lớn (tương đương 32) nhưng chỉ tốn VRAM của Batch Size 8. Rất hữu ích trên phần cứng giới hạn (ví dụ: Kaggle GPUs). |
| **Hàm Mất Mát (Loss Function)** | Cross-Entropy (Softmax thuần) | **NCOD Bi-level Optimization (L1 + L2)** | (QUAN TRỌNG NHẤT) Xử lý nhãn nhiễu (noisy labels) tự nhiên trong CausalVidQA. Thay vì tin mù quáng vào Ground Truth, loss function tính thêm tham số `U` để "discount" các câu hỏi gây bối rối. |
| **Hệ Thống Optimizer** | 1 Optimizer duy nhất (Adam hoặc AdamW) | **Dual Optimizers** (AdamW cho Model, SGD cho cơ chế `U`) | Tối ưu hóa song song (Bi-level). Model học features, SGD tuning tham số `U` riêng biệt (lr=0.1) để bắt lỗi nhanh chóng. |
| **Checkpoint & State Tracking** | Chỉ lưu Weights Model Epoch cuối hoặc Best | **Lưu Toàn Bộ State** (Weights, `U` vector, Optimizers, Scheduler) | Hỗ trợ Resume Training mượt mà. Không lo mất trạng thái của Adaptive LR (Adam/Scheduler) hoặc của vec-tơ `U`. |
| **Dropout Regularization** | `0.1` (Thường thấy trong bản gốc) | **`0.3`** | Tăng cường độ regularization để bù trừ cho capacity lớn của DINOv3, tránh overfitting quá nhanh trên tập training. |

---

## 2. Đi Sâu Vào Thay Đổi Backbone: DINOv3 vs ResNet Duality

### 2.1 Kiến trúc Gốc (ResNet Duality)
- **Luồng dữ liệu**: Frame gốc → ResNet-101 (trích xuất diện mạo - appearance) & 3D-ResNet (trích xuất chuyển động - motion).
- **Phép nối**: `Concat(Appearance, Motion)`.
- **Hạn chế**:
  - `4096` chiều chứa nhiều thông tin rời rạc. Bản chất ResNet-101 (supervised trên ImageNet) rấy nhạy với các object classes cụ thể, nhưng kém trong việc tóm tắt "hoàn cảnh" / "context" cần thiết cho Causal Reasoning (Suy luận nhân quả).
  - 3D-ResNet tốn kém lưu trữ và không phải lúc nào chuyển động cũng liên quan trực tiếp đến các nhóm câu hỏi Predictive (Dự đoán) hoặc Counterfactual (Giả định).

### 2.2 Kiến trúc Hiện Tại (DINOv3 ViT-L/14)
- **Luồng dữ liệu**: Frame gốc → DINOv3 ViT-L/14 → Lấy `[CLS] token` cho mỗi frame.
- **Tại sao hiệu quả hơn?**:
  - DINOv3 được huấn luyện Self-Supervised trên `LVD-142M` (142 triệu ảnh). Thay vì chỉ nhận diện "con mèo", nó học được "sự vật A đang tương tác với sự vật B trong không gian C". Nghĩa là, semantic space của DINOv3 **tương đồng hơn** với không gian ngôn ngữ mà mô hình DeBERTa đang hiểu.
  - Phù hợp hơn cho module `VLEncoder` và `FrameDecoder` của TranSTR trong việc kết hợp Text <-> Visual.

---

## 3. Bước Đột Phá: NCOD (Noisy Correspondence Detection)

Đây là thay đổi cấu trúc ở cấp độ **Meta-Learning / Loss Function**, khác hoàn toàn so với TranSTR gốc.

### 3.1 TranSTR Gốc (Naive Cross Entropy)
- Hoạt động theo logic: Nhãn (Ground Truth) luôn đúng.
- `L = CrossEntropy(Logits, Label)`.
- **Hậu quả trên CausalVidQA**: Dataset này chứa các câu hỏi như "Tại sao người đàn ông lại chạy?". Có 5 đáp án, trong đó "Tập thể dục" và "Sắp trễ xe buýt" đều có thể là đáp án hợp lý nếu nhìn bằng mắt thường. Khi ép model học 1 nhãn duy nhất, model sẽ bị nhiễu (overfit vào noise), dẫn đến Validation Accuracy tạo hình parapol ngược (đạt đỉnh sớm rồi giảm).

### 3.2 TranSTR-DN (Bi-level Optimization với chiết khấu `U`)
- Chèn thêm một **Vec-tơ U** (khởi tạo nhỏ ~1e-8, chiều dài bằng tổng số samples trong tập train).
- **Cập nhật Mô Hình (L1)**:
  - `p_shifted = Softmax(Logits) + U[i].detach() * Label`
  - Nếu `U[i]` cao (hệ thống nghi ngờ đây là câu hỏi nhiễu), `p_shifted` tại nhãn đúng sẽ được "bơm" lên gần 1.0. Loss L1 sẽ cực nhỏ $\rightarrow$ **Mô hình lướt qua, không học mẫu này.**
- **Cập nhật Vec-tơ U (L2)**:
  - Tách biệt (`detach`) xác suất của mô hình. Tối ưu hóa `U` sao cho nó tiệm cận với khoảng cách sai số giữa Mô Hình và Nhãn.
  - Sử dụng SGD thuần, Learning Rate = 0.1 (nhanh vọt).

Kiến trúc này biến TranSTR từ một mô hình "nạn nhân của dataset" thành một mô hình **chủ động phân loại dữ liệu sạch và dữ liệu bẩn ngay trong lúc huấn luyện.**

---

## 4. Các Tối Ưu Hóa Kỹ Thuật (Engineering) Khác

1. **Dual Optimizers**: TranSTR gốc chỉ dùng 1 Adam optimizer cho toàn bộ mạng. Hiện tại, chúng ta tách `opt_model` (AdamW cho Neural Net) và `opt_U` (SGD 1 chiều cho tham số chiết khấu).
2. **Gradient Accumulation**: Cho phép chạy `bs=8` x `4 accum` một cách trong suốt. Loss L1 và L2 đều được lấy trung bình (`scaled_loss = L / 4`) để đảm bảo norm của gradient không bị phình to. Điều này khắc phục hạn chế VRAM của TranSTR gốc.
3. **W&B Integration Sâu Hơn**: Logger hiện tại bắt (watch) histogram của phân phối `U` mỗi epoch. Qua đó:
   - Nếu `U` nhô lên ở hai đỉnh (bimodal) -> Cơ chế chống nhiễu đang hoạt động hoàn hảo.
   - Cho phép query thẳng `top_k` câu hỏi ồn ào nhất đẩy lên WANDB table. Tăng tính minh bạch (Explainability) cho mô hình - thứ mà TranSTR gốc không có.
4. **Resumable State**: Việc nạp lại toàn bộ `scheduler.state_dict()`, `opt_model.state_dict()` giúp luồng Learning Rate decay không bị "gãy" khi người dùng cần huấn luyện tiếp từ epoch thứ 7.

---
**Kết Luận:** Về mặt kiến trúc mạng Neural thuần túy (các lớp Attention, Transformer Decoder), chúng ta giữ nguyên bố cục TranSTR gốc để chứng minh tính hiệu quả của phương pháp. Sự lột xác nằm ở **Chất lượng đầu vào** (DINOv3) và **Cơ chế chống nhiễu đầu ra** (NCOD).
