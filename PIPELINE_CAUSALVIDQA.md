# Quy Trình Pipeline CausalVidQA Zero-Shot Qwen-VL

## Tổng Quan

Pipeline này thực hiện **zero-shot video question answering** sử dụng mô hình Qwen-VL trên bộ dữ liệu CausalVidQA với cấu trúc chia `train/val/test`. Mô hình dự đoán câu trả lời dựa vào 16 frame video được lấy mẫu đều nhau.

---

## 1. Cài Đặt và Chuẩn Bị Dữ Liệu

### 1.1 Cài Thư Viện
```bash
pip install -q transformers accelerate decord opencv-python pillow pandas tqdm
```

### 1.2 Cấu Hình Đường Dẫn

| Biến | Mô Tả |
|------|-------|
| `VIDEO_ROOT` | Thư mục chứa các file video |
| `ANNOTATION_ROOT` | Thư mục chứa dữ liệu chú thích (text.json, answer.json) theo video_id |
| `SPLIT_DIR` | Thư mục chứa file phân chia: `train.pkl/txt`, `valid.pkl/txt`, `test.pkl/txt` |

### 1.3 Cấu Hình Mô Hình
- **Mô hình**: `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen VL 3B)
- **Số frame/video**: 16 frame
- **Max tokens sinh ra**: 8 token
- **Thiết bị**: CUDA nếu có, nếu không dùng CPU

---

## 2. Cấu Trúc Dữ Liệu

### 2.1 Kiểu Câu Hỏi (Question Types)
Pipeline hỗ trợ 4 kiểu câu hỏi chính:
- `descriptive` - Câu hỏi mô tả
- `explanatory` - Câu hỏi giải thích
- `predictive` - Câu hỏi dự đoán
- `counterfactual` - Câu hỏi phản thực

### 2.2 Cấu Trúc File Chú Thích
Mỗi video_id có hai file JSON:

**text.json** - Chứa câu hỏi và các đáp án
```json
{
  "descriptive": {
    "question": "...",
    "answer": ["option0", "option1", "option2", "option3", "option4"]
  },
  "predictive": {
    "question": "...",
    "answer": ["...", "...", "...", "...", "..."],
    "reason": ["reason0", "reason1", "reason2", "reason3", "reason4"]
  },
  ...
}
```

**answer.json** - Chứa đáp án đúng
```json
{
  "descriptive": {
    "answer": 2  // chỉ số đáp án đúng (0-4)
  },
  "predictive": {
    "answer": 1,
    "reason": 3
  },
  ...
}
```

### 2.3 Phân Chia Dữ Liệu
- `train.pkl/txt` - Tập huấn luyện
- `valid.pkl/txt` hoặc `val.pkl/txt` - Tập xác thực
- `test.pkl/txt` - Tập kiểm tra

Mỗi file chứa danh sách video_id (một dòng/một entry)

---

## 3. Quy Trình Xử Lý

### 3.1 Đọc Dữ Liệu Split

**Input**: Split name (train/val/test)  
**Output**: Danh sách video_id

```
Split file (.pkl hoặc .txt)
        ↓
    Parse video_id
        ↓
    Sorted list of video_id
```

### 3.2 Xây Dựng Records

**Input**: Danh sách video_id  
**Output**: Danh sách record (câu hỏi + đáp án)

Với mỗi video_id:
1. Đọc `text.json` từ `ANNOTATION_ROOT/video_id/text.json`
2. Đọc `answer.json` từ `ANNOTATION_ROOT/video_id/answer.json`
3. Với mỗi question type:
   - Trích xuất: question, candidates (5 lựa chọn), ground-truth index
   - Tạo record: `{video_id, qtype, question, candidates, gt}`
   - Với predictive/counterfactual: tạo thêm record cho reason

### 3.3 Tìm File Video

**Input**: video_id  
**Output**: Đường dẫn video file

Quy trình tìm kiếm:
1. Thử tìm file trực tiếp: `VIDEO_ROOT/video_id.ext` (ext ∈ [.mp4, .avi, .mov, .mkv, .webm])
2. Nếu không tìm thấy, tìm recursive trong `VIDEO_ROOT`
3. Return `None` nếu không tìm thấy

### 3.4 Lấy Mẫu Frame

**Input**: Video path, số frame = 16  
**Output**: Danh sách 16 ảnh PIL.Image

```
Video file (cv2.VideoCapture)
    ↓
Đếm tổng frame (total)
    ↓
Tính indices: linspace(0, total-1, 16)
    ↓
Đọc frame tại mỗi index
    ↓
Convert BGR → RGB
    ↓
Trả về danh sách PIL.Image
```

### 3.5 Load Mô Hình

```
Qwen2.5-VL-3B-Instruct
    ↓
dtype: bfloat16 (nếu CUDA) hoặc float32 (CPU)
    ↓
device_map: 'auto' (CUDA) hoặc 'cpu'
    ↓
model.eval()
```

### 3.6 Xây Dựng Prompt

**Input**: question, candidates (danh sách 5 đáp án)  
**Output**: Prompt text

```
Template:
You are a video question answering assistant.
Watch the video frames and choose the best option index from 0 to 4.
Question: [question]
Options:
0. [candidate 0]
1. [candidate 1]
2. [candidate 2]
3. [candidate 3]
4. [candidate 4]
Answer with one digit only: 0, 1, 2, 3, or 4.
```

### 3.7 Dự Đoán (Inference)

**Input**: Record (question, candidates, video_id)  
**Output**: Predicted answer index, Raw model output

Quy trình:
```
Record
  ↓
Tìm video file
  ↓
Lấy mẫu 16 frame
  ↓
Xây dựng prompt
  ↓
Format message (role: user, content: [video, text])
  ↓
Apply chat template
  ↓
Processor: text + video → input tensors
  ↓
Qwen model.generate (max_new_tokens=8)
  ↓
Decode output text
  ↓
Parse digit (0-4) từ output
  ↓
Return (predicted_index, raw_text)
```

### 3.8 Đánh Giá (Evaluation)

**Input**: Split name, Max samples (tuỳ chọn)  
**Output**: DataFrame kết quả, Statistics dict

Với mỗi record:
1. Gọi `predict_one(record)`
2. So sánh: `pred == gt` → correct
3. Tính accuracy: `(correct / parsed) × 100%`

**Metrics**:
- `records`: tổng số record
- `parsed`: số record có dự đoán hợp lệ (0-4)
- `acc`: accuracy (%)

---

## 4. Quy Trình Chính

### 4.1 Luồng Thực Thi

```
START
  ↓
For split in [train, val, test]:
  ├─→ Đọc split IDs
  ├─→ Build records cho mỗi video
  ├─→ For mỗi record:
  │   ├─→ Tìm video file
  │   ├─→ Lấy 16 frame
  │   ├─→ Xây dựng prompt
  │   ├─→ Inference với Qwen-VL
  │   ├─→ Parse dự đoán
  │   └─→ Ghi kết quả
  ├─→ Tính accuracy
  └─→ Lưu results

Tổng hợp results từ 3 split
  ↓
Lưu summary.csv + predictions.csv
  ↓
END
```

### 4.2 Output

**summary.csv**: Tóm tắt theo split
```
split | records | parsed | acc
------|---------|--------|-----
train | 450     | 445    | 78.43
val   | 100     | 98     | 72.45
test  | 150     | 148    | 75.34
```

**predictions.csv**: Chi tiết mỗi dự đoán
```
split | video_id | qtype       | question              | gt | pred | correct | raw_output
------|----------|-------------|----------------------|----|----- |---------|------------
train | vid_001  | descriptive | What is in the scene?| 2  | 2    | True    | "2"
train | vid_002  | predictive  | What will happen?    | 1  | 3    | False   | "Answer: 3"
...
```

---

## 5. Cấu Hình Tuỳ Chỉnh

| Tham Số | Giá Trị Mặc Định | Mô Tả |
|---------|-----------------|-------|
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Mô hình sử dụng |
| `NUM_FRAMES` | 16 | Số frame lấy mẫu từ video |
| `MAX_NEW_TOKENS` | 8 | Số token tối đa sinh ra |
| `MAX_SAMPLES_PER_SPLIT` | None | Số mẫu tối đa/split (None = dùng hết) |
| `DEVICE` | CUDA/CPU (tự động) | Thiết bị tính toán |

---

## 6. Xử Lý Lỗi và Ngoại Lệ

| Lỗi | Nguyên Nhân | Xử Lý |
|-----|-----------|------|
| `video_not_found` | Video file không tìm thấy | Bỏ qua record, không tính vào accuracy |
| `no_frames` | Không đọc được frame nào | Bỏ qua record |
| Parse error (chữ không phải 0-4) | Mô hình không sinh đầu ra đúng định dạng | `pred = None`, không tính vào accuracy |
| JSON decode error | File JSON bị lỗi | Bỏ qua video đó |

---

## 7. Điểm Chú Ý

1. **Sampling Frame Đều Nhau**: Sử dụng `np.linspace` để đảm bảo frame được lấy mẫu đều trong toàn bộ video
2. **Chia Batch Processing**: Dùng `tqdm` để tracking tiến độ
3. **Tokenization Format**: Qwen yêu cầu message format cụ thể với role/content
4. **Device Management**: `torch.cuda` tự động nếu CUDA có sẵn
5. **Regex Extraction**: Parse digit từ output text bằng `re.search(r'[0-4]', text)`

---

## 8. Yêu Cầu Tối Thiểu

- **GPU Memory**: ~6GB (cho 3B model)
- **Disk**: Đủ cho video files (phụ thuộc dataset)
- **Python**: 3.8+
- **CUDA**: 11.0+ (tuỳ chọn, CPU cũng chạy được nhưng chậm)

---

## 9. Ví Dụ Chạy

```python
# Chạy toàn bộ 3 split
for split_name in ['train', 'val', 'test']:
    df_split, stats = evaluate_split(split_name)
    # df_split: DataFrame kết quả
    # stats: {'split': ..., 'records': ..., 'parsed': ..., 'acc': ...}

# Lưu kết quả
all_dfs → predictions.csv
summary_df → summary.csv
```

---

**Lần cập nhật cuối**: 2026-05-12  
**Notebook**: `ZeroShot_Qwen_Video.ipynb`
