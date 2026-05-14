# TranSTR for CausalVidQA

Triển khai training/evaluation cho bài toán Video Question Answering trên CausalVidQA, dựa trên TranSTR.

## 1) Tổng quan

- Task: chọn 1 trong 5 đáp án cho mỗi câu hỏi từ video.
- Dữ liệu gồm 6 loại câu hỏi: `descriptive`, `explanatory`, `predictive answer/reason`, `counterfactual answer/reason`.
- Pipeline chính trong repo:
  - Load visual features + annotation + split
  - Train/eval model TranSTR
  - Xuất checkpoint và file prediction

## 2) Cấu trúc repo quan trọng

```text
causalvid/
├── DataLoader.py
├── train.py
├── test.py
├── eval_mc.py
├── networks/
│   └── model.py
├── utils/
├── trainingtranstr.ipynb
└── README.md
```

## 3) Input dataset

Bạn có thể dùng Kaggle datasets hoặc đường dẫn local. Code hiện tại trong notebook `trainingtranstr.ipynb` đang dùng KaggleHub để tự resolve đường dẫn.

### 3.1 Dataset links (điền/tùy chỉnh)

- Video/frame features: `danielq07/vit-features-full-merged`
- Object features: `danielq07/object-detection-causal-full`
- Text annotation: `lusnaw/text-annotation`
- Split files: `danielq07/casual-vid-data-split`

Tập dữ liệu gốc lấy từ youtube dùng cho việc trích xuất lại các đặc trưng:
- Raw videos (`video_id.mp4`): `danielq07/causal-vidqa-raw-video-full`

### 3.2 Cấu trúc dữ liệu kỳ vọng

```text
<video_feature_root>/
└── *.pt

<object_feature_root>/
└── *.pkl (hoặc subfolder chứa *.pkl)

<annotation_root>/QA/
└── <video_id>/
    ├── text.json
    └── answer.json

<split_root>/split/
├── train.pkl
├── valid.pkl
└── test.pkl
```

## 4) Weight model 

### 4.1 Pretrained / trained weights

- Best checkpoint link: `https://drive.google.com/file/d/1yHcW-DNo_xhGbjCdVNagpS6ILbZBC5r1/view?usp=sharing`
- Last checkpoint link: `https://drive.google.com/drive/folders/1DHcD8v1p0MBW-n-G35K9IqGdGIn5Klk-?usp=sharing`

### 4.2 Vị trí đặt weight local

Đặt file vào một trong các path sau để notebook/script tự load:

```text
/kaggle/working/models/best_model.ckpt
./models/best_model.ckpt
```

## 5) Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Nếu chạy notebook Kaggle/Colab, có thể cần thêm:

```bash
pip install -U kagglehub wandb huggingface_hub
```

## 6) Cách dùng

### 6.1 Cách 1: dùng notebook (khuyến nghị)

Notebook chính: `tranSTR.ipynb`
Link colab mẫu: `https://colab.research.google.com/drive/1VvCL711old__GnPnazBoXVN2yZ39ULhl?usp=sharing`
Notebook inference: `inference.ipynb`
Link kaggle inference mẫu: `https://www.kaggle.com/code/thnhinhoquc/inference-transtr`

Flow:
- Cell setup + auth
- Cell resolve dataset paths
- Tạo dataset/train-val-test loaders
- Train model + save checkpoint
- Evaluate test set

Single-video inference:
- Trong `inference.ipynb`, chạy thêm `CELL 7: Single-video inference`.
- Đặt `VIDEO_ID` (vd `2VBmRPrfNZY_000000_000010`).
- Cell sẽ tự download raw video từ dataset `danielq07/causal-vidqa-raw-video-full`, hiển thị video, và xuất kết quả dự đoán riêng cho video đó.

### 6.2 Cách 2: chạy bằng command line

Train:

```bash
python train.py \
  -v transtr_run \
  -bs 16 \
  -lr 1e-4 \
  -epoch 20 \
  -gpu 0 \
  --sample_list_path "/path/to/QA_or_split" \
  --video_feature_path "/path/to/video_features" \
  --text_annotation_path "/path/to/text-annotation" \
  --qtype -1
```

Eval/Test:

```bash
python test.py \
  -v transtr_eval \
  -bs 32 \
  --sample_list_path "/path/to/QA_or_split" \
  --video_feature_path "/path/to/video_features" \
  --text_annotation_path "/path/to/text-annotation" \
  --qtype -1 \
  --model_path "./models/best_model.ckpt"
```

## 7) Output

```text
models/
├── best_model.ckpt
└── last_checkpoint.ckpt

prediction/
└── *.json

log/
└── *.log
```

## 8) Lưu ý thường gặp

- Nếu lỗi `fp16` với DeBERTa: tắt AMP (`use_amp=False`).
- Nếu lỗi `num_workers` trên Kaggle: đặt `num_workers=0`.
- Nếu split train bất thường: kiểm tra lại file `train.pkl` và dataset split đang dùng.

## 9) References

- [IGV Paper (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
- [CausalVidQA Dataset](https://github.com/bcmi/Causal-VidQA)
- [Original IGV Code](https://github.com/yl3800/IGV)

## 10) Citation

```bibtex
@InProceedings{Li_2022_CVPR,
    author    = {Li, Yicong and Wang, Xiang and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title     = {Invariant Grounding for Video Question Answering},
    booktitle = {CVPR},
    year      = {2022},
    pages     = {2928-2937}
}
```