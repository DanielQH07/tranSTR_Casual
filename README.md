<h2 align="center">
Invariant Grounding for Video Question Answering 🔥
</h2>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/yl3800/IGV) 
[![](https://img.shields.io/badge/video-red?style=plastic&logo=airplayvideo)](https://youtu.be/wJhR9_dcsaM) 
</div>


## Overview 
This repo contains source code for **Invariant Grounding for Video Question Answering** (CVPR 2022 Oral, Best Paper Finalists). In this work, propose a new learning framework, Invariant Grounding for VideoQA (**IGV**), to ground the question-critical scene, whose causal relations with answers are invariant across different interventions on the complement. With IGV, the VideoQA models are forced to shield the answering process from the negative influence of spurious correlations, which significantly improves the reasoning ability.
    
<figure> <img src="figures/interventional-distributions.png" height="220"></figure>

---

# CausalVidQA - Training Guide

## 📁 Cấu trúc dữ liệu

Dữ liệu CausalVidQA được tải từ Kaggle:

```
visual-feature/
├── appearance_feat.h5    # Appearance features (ResNet)
├── motion_feat.h5        # Motion features (ResNet)
└── idx2vid.pkl           # Video ID mapping

text-annotation/
├── video_id_1/
│   ├── text.json         # Questions và candidate answers
│   └── answer.json       # Ground truth answers
├── video_id_2/
│   └── ...

dataset-split-1/
├── train.pkl             # Train video IDs
├── val.pkl               # Validation video IDs
└── test.pkl              # Test video IDs
```

## 🔧 Cài đặt

```bash
pip install -r requirements.txt
pip install kagglehub
```

## 📥 Download dữ liệu

```python
import kagglehub

text_feature_path = kagglehub.dataset_download('lusnaw/text-feature')
visual_feature_path = kagglehub.dataset_download('lusnaw/visual-feature')
split_path = kagglehub.dataset_download('lusnaw/dataset-split-1')
text_annotation_path = kagglehub.dataset_download('lusnaw/text-annotation')
```

## 🚀 Training

### Train đầy đủ

```bash
python train.py \
    -v full_train \
    -bs 32 \
    -lr 1e-5 \
    -epoch 15 \
    -gpu 0 \
    --sample_list_path "/path/to/dataset-split-1" \
    --video_feature_path "/path/to/visual-feature" \
    --text_annotation_path "/path/to/text-annotation" \
    --qtype -1 \
    -fk 8 \
    -ok 5 \
    -objs 20 \
    -el 1 \
    -dl 1 \
    -t microsoft/deberta-base
```

### Train nhanh (test với số video giới hạn)

```bash
# Train với 10 videos (60 samples vì mỗi video có 6 loại câu hỏi)
python train.py \
    -v quick_test \
    -bs 4 \
    -lr 1e-4 \
    -epoch 2 \
    -gpu 0 \
    --sample_list_path "/path/to/dataset-split-1" \
    --video_feature_path "/path/to/visual-feature" \
    --text_annotation_path "/path/to/text-annotation" \
    --qtype -1 \
    --max_samples 10 \
    -fk 4 \
    -ok 5 \
    -objs 10
```

### Train theo loại câu hỏi cụ thể

```bash
# Chỉ train với câu hỏi descriptive (qtype=0)
python train.py -v descriptive_only --qtype 0 ...
```

## 🧪 Testing

```bash
python test.py \
    -v test_eval \
    -bs 32 \
    -gpu 0 \
    --sample_list_path "/path/to/dataset-split-1" \
    --video_feature_path "/path/to/visual-feature" \
    --text_annotation_path "/path/to/text-annotation" \
    --qtype -1 \
    -fk 8 \
    -ok 5 \
    -objs 20 \
    -t microsoft/deberta-base \
    --model_path "./models/best_model-xxx.ckpt"
```

## 🎯 Script chạy nhanh

```bash
# Tự động download data và train
python run_small_test.py --run

# Train với số video tùy chỉnh
python run_small_test.py --run --max_samples 50
```

## 📋 Tham số chính

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `-v` | Tên version/experiment | (required) |
| `-bs` | Batch size | 32 |
| `-lr` | Learning rate | 1e-5 |
| `-epoch` | Số epochs | 15 |
| `-gpu` | GPU ID | 0 |
| `--qtype` | Loại câu hỏi (-1=all, 0-5=specific) | -1 |
| `--max_samples` | Giới hạn số video | None (all) |
| `-fk` | Top-K frames | 8 |
| `-ok` | Top-K objects | 5 |
| `-objs` | Số objects per frame | 20 |
| `-el` | Encoder layers | 1 |
| `-dl` | Decoder layers | 1 |
| `-t` | Text encoder model | microsoft/deberta-base |

## 📊 Loại câu hỏi (qtype)

| qtype | Loại câu hỏi | Mô tả |
|-------|--------------|-------|
| -1 | All | Tất cả 6 loại |
| 0 | Descriptive | Mô tả |
| 1 | Explanatory | Giải thích |
| 2 | Predictive Answer | Dự đoán (câu trả lời) |
| 3 | Predictive Reason | Dự đoán (lý do) |
| 4 | Counterfactual Answer | Phản thực (câu trả lời) |
| 5 | Counterfactual Reason | Phản thực (lý do) |

## 📂 Output

- **Models**: `./models/best_model-{version}.ckpt`
- **Predictions**: `./prediction/{version}-{epoch}-{acc}.json`
- **Logs**: `./log/{version}.log`

## 💡 Ví dụ Windows PowerShell

```powershell
cd d:\KLTN\TranSTR\causalvid

# Train với 10 videos
python train.py -v test10 -bs 4 -epoch 2 -gpu 0 `
    --sample_list_path "C:\Users\xxx\.cache\kagglehub\datasets\lusnaw\dataset-split-1\versions\1" `
    --video_feature_path "C:\Users\xxx\.cache\kagglehub\datasets\lusnaw\visual-feature\versions\1" `
    --text_annotation_path "C:\Users\xxx\.cache\kagglehub\datasets\lusnaw\text-annotation\versions\1" `
    --max_samples 10 -fk 4 -ok 5 -objs 10
```

## 🔍 Evaluation Metrics

Kết quả được đánh giá theo từng loại câu hỏi:

- **Des**: Descriptive accuracy
- **Exp**: Explanatory accuracy  
- **Pred-A**: Predictive Answer accuracy
- **Pred-R**: Predictive Reason accuracy
- **CF-A**: Counterfactual Answer accuracy
- **CF-R**: Counterfactual Reason accuracy
- **Pred**: Predictive (cả answer và reason đúng)
- **CF**: Counterfactual (cả answer và reason đúng)
- **ALL**: Overall accuracy (Des + Exp + Pred + CF)

---

## Installation (Original)
- Main packages: PyTorch = 1.11 
- See `requirements.txt` for other packages.

## Data Preparation (Original)
We use MSVD-QA as an example to help get farmiliar with the code. Please download the pre-computed features and trained models [here](https://drive.google.com/file/d/1MrupFq8jubEA4nEl4CppR5Rddz9rW_6Z/view?usp=sharing)

After downloading the data, please modify your data path in `run.py`.

## Run IGV

Simply run `run.sh` to reproduce the results in the paper. 


## Reference 
```
@InProceedings{Li_2022_CVPR,
    author    = {Li, Yicong and Wang, Xiang and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title     = {Invariant Grounding for Video Question Answering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2928-2937}
}
```