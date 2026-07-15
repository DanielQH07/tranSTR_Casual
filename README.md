# TranSTR-DN — CausalVidQA

TranSTR run1 dùng DINOv3 frame features, GroundingDINO/Faster R-CNN object features,
DeBERTa-base, NCOD/LUM/HUM, verifier loss, hard-negative weighting và EMA.

**Nguồn cấu hình duy nhất:** [`config.yaml`](./config.yaml). `train.py` và `test.py` chỉ nhận
đường dẫn YAML; không chỉnh hyperparameter trực tiếp trong hai script.

## Tệp chính / Main files

| Tệp | Mục đích / Purpose |
|---|---|
| `config.yaml` | Cấu hình chung cho train và test / shared train-test configuration |
| `train.py` | Train local bằng cấu hình YAML / local YAML-driven training |
| `test.py` | Load checkpoint, inference và xuất CSV / checkpoint evaluation and CSV export |
| `download_weight.py` | Tải checkpoint từ KaggleHub vào `weights/` |
| `example/train_colab.ipynb` | Notebook train Colab, batch 32 |
| `example/inference_colab.ipynb` | Notebook inference Colab và xuất CSV đầy đủ |
| `example/inference_kaggle.ipynb` | Notebook Kaggle inference toàn bộ test video, đối chiếu raw MP4 và xuất CSV |
| `example/train_kaggle.ipynb` | Notebook Kaggle dựa trên notebook n2 gốc |

---

# Tiếng Việt

## 1. Cài đặt và dữ liệu

Yêu cầu Python có PyTorch/CUDA phù hợp, sau đó cài dependency:

```bash
pip install -r requirements.txt
```

Tải và giải nén bốn dataset đầu vào model. Dataset raw MP4 dùng khi tự sinh feature hoặc chạy notebook inference Kaggle:

| Thành phần | Kaggle dataset | Thư mục local |
|---|---|---|
| DINOv3 frames | [danielq07/dinov3-feat](https://www.kaggle.com/datasets/danielq07/dinov3-feat) | `data/visual_feature/` |
| GDINO + FRCNN objects | [danielq07/causal-vidqa-gdinofasterrcnn-features-merged](https://www.kaggle.com/datasets/danielq07/causal-vidqa-gdinofasterrcnn-features-merged) | `data/object_feature/` |
| QA annotations | [lusnaw/text-annotation](https://www.kaggle.com/datasets/lusnaw/text-annotation) | `data/text_annotation/` |
| Data splits | [danielq07/casual-vid-data-split](https://www.kaggle.com/datasets/danielq07/casual-vid-data-split) | `data/data_split/` |
| Raw MP4 (sinh feature/đối chiếu inference) | [danielq07/causal-vidqa-raw-video-full](https://www.kaggle.com/datasets/danielq07/causal-vidqa-raw-video-full) | Không bắt buộc cho `train.py`; KaggleHub resolve tự động |

Cấu trúc bắt buộc:

```text
data/
├── visual_feature/<video_id>.pt          # tất cả split nằm phẳng
├── object_feature/<video_id>.pkl         # nằm phẳng
├── text_annotation/QA/<video_id>/
│   ├── text.json
│   └── answer.json
└── data_split/split/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

`DataLoader.py` không quét đệ quy frame/object feature. Nếu DINOv3 được chia thành
`train/valid/test`, phải move hoặc symlink toàn bộ `.pt` vào `data/visual_feature/`.

## 2. Cấu hình hiện tại

| Nhóm | Giá trị hiệu lực |
|---|---|
| Run | `run1_colab_bs32_ncod_hum_verifier`, seed `999`, bắt buộc CUDA |
| Batch | Physical `32`, accumulation `1`, effective `32` |
| DataLoader | `num_workers=0`, `pin_memory=false` |
| Frame | Input `[16,1024]`, model chọn `5` frame |
| Object | `[16,12,2820]` = `2048 ROI + 768 class + 4 bbox` |
| Model | `d_model=768`, 8 heads, 2 encoder layers, dropout `0.3` |
| Text | `microsoft/deberta-base`, full fine-tuning |
| Optimizer | AdamW; main LR `1e-5`, text LR `5e-6`, decay `1e-4` |
| Scheduler | Cosine, warmup 1 epoch, tối đa 10 epochs |
| Loss | NCOD/LUM/HUM + verifier `0.25`; verifier tắt ở epoch 1–2 |
| Regularization | Hard negative max `1.5`, EMA `0.999`, gradient clip `1.0` |
| Early stop | Bắt đầu epoch 5, patience 4, delta `0.05` |
| Knowledge | Tắt trong pipeline `train.py/test.py`; evaluation dùng `out['logits']` |

Object collator pad/truncate về `[16,12,2820]` và thay NaN/Inf bằng 0. GroundingDINO dùng
toàn bộ 12 object của các frame được chọn. `question_family_id` chỉ route LUM/HUM bên ngoài
model; knowledge modules bị gỡ trước optimizer, EMA và checkpoint.

## 3. Weight và inference

`config.yaml` hiện trỏ tới:

```text
weights/best_model_gdinofrcnn_ncod_hum_run1_generic_safe_lora_hn_ema_cos.ckpt
```

Nguồn weight: `danielq07/gdinofrcnn-ncod-hum-model`. Sau khi cấu hình `KAGGLE_API_TOKEN`
và được cấp quyền dataset, chạy:

```bash
python download_weight.py
python test.py --config config.yaml
```

`test.py` ưu tiên `ema_model_state_dict`; nếu checkpoint không có EMA thì dùng
`model_state_dict`. Muốn đổi weight, chỉ sửa `test.checkpoint_path` trong YAML.

Notebook `example/inference_kaggle.ipynb` tải trực tiếp weight trên, toàn bộ raw MP4 từ
`danielq07/causal-vidqa-raw-video-full`, chạy hết test split với batch 32/workers 0 và thêm
`raw_video_path`, `raw_video_exists` vào CSV. Model vẫn dùng feature DINOv3/GDINO dựng sẵn.

## 4. Train, kiểm tra và output

```bash
# Chỉ kiểm tra YAML, không tải data/model
python train.py --config config.yaml --check-config
python test.py --config config.yaml --check-config

# Chạy thật
python train.py --config config.yaml
python test.py --config config.yaml
```

Output nằm trong `outputs/models/`: best/last checkpoint, history CSV, prediction CSV,
metrics JSON và comparison CSV. W&B dùng `WANDB_API_KEY` khi `mode: online`; có thể đổi thành
`offline` hoặc `disabled`. Weight chỉ được upload một lần ở cuối run: một `last` và một `best`.

Metric chính:

```text
PAR = Predictive-Answer và Predictive-Reason cùng đúng trên một video
CAR = Counterfactual-Answer và Counterfactual-Reason cùng đúng trên một video
Acc_ALL = (Description + Explanation + PAR + CAR) / 4
```

> `example/train_kaggle.ipynb` giữ knowledge loss và memory post-check của notebook n2 gốc;
> đây là biến thể khác với pipeline YAML hiện tại đã tắt knowledge.

---

# English

## 1. Installation and data

Use a CUDA-compatible Python/PyTorch environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Download the same four model-input datasets listed above and extract them into this flat layout. The full raw MP4 dataset is [danielq07/causal-vidqa-raw-video-full](https://www.kaggle.com/datasets/danielq07/causal-vidqa-raw-video-full); it is used for feature extraction and raw-video coverage in the Kaggle inference notebook, but is not read directly by `train.py`:

```text
data/
├── visual_feature/<video_id>.pt
├── object_feature/<video_id>.pkl
├── text_annotation/QA/<video_id>/{text.json,answer.json}
└── data_split/split/{train.pkl,valid.pkl,test.pkl}
```

Frame `.pt` and object `.pkl` files must be directly inside their configured directories;
`DataLoader.py` does not scan those feature directories recursively.

## 2. Current configuration

| Group | Effective value |
|---|---|
| Run | `run1_colab_bs32_ncod_hum_verifier`, seed `999`, CUDA required |
| Batch | Physical `32`, accumulation `1`, effective `32` |
| DataLoader | `num_workers=0`, `pin_memory=false` |
| Frames | Input `[16,1024]`, select `5` |
| Objects | `[16,12,2820]` = `2048 ROI + 768 class + 4 bbox` |
| Model | `d_model=768`, 8 heads, 2 encoder layers, dropout `0.3` |
| Text | `microsoft/deberta-base`, full fine-tuning |
| Optimization | AdamW; main LR `1e-5`, text LR `5e-6`, decay `1e-4` |
| Training | 10 epochs; cosine schedule; one warmup epoch |
| Loss | NCOD/LUM/HUM + verifier `0.25`; verifier off in epochs 1–2 |
| Regularization | Hard-negative max `1.5`, EMA `0.999`, clip `1.0` |
| Early stop | Start epoch 5, patience 4, delta `0.05` |
| Knowledge | Disabled in `train.py/test.py`; evaluation uses `out['logits']` |

The collator pads/truncates object tensors to `[16,12,2820]` and sanitizes NaN/Inf. Family IDs
route LUM/HUM outside the model; knowledge modules are removed before optimizer/EMA/checkpoint
creation.

## 3. Weight, train, and inference

The configured test checkpoint is:

```text
weights/best_model_gdinofrcnn_ncod_hum_run1_generic_safe_lora_hn_ema_cos.ckpt
```

After granting Kaggle access and configuring `KAGGLE_API_TOKEN`:

```bash
python download_weight.py
python test.py --config config.yaml
```

Validate or train with:

```bash
python train.py --config config.yaml --check-config
python test.py --config config.yaml --check-config
python train.py --config config.yaml
```

Outputs are written to `outputs/models/`. Online W&B requires `WANDB_API_KEY`; set the YAML
mode to `offline` or `disabled` when uploads are not needed. Training uploads exactly one
`last` and one `best` weight artifact at finish.

`example/inference_kaggle.ipynb` downloads the fixed Kaggle checkpoint and full raw MP4 dataset,
runs the complete test split with batch 32/workers 0, verifies raw-video coverage, and adds the
resolved MP4 path to the output CSV. TranSTR inference still consumes precomputed DINOv3/GDINO features.

> `example/train_kaggle.ipynb` intentionally retains the original n2 knowledge loss and memory
> post-check; it is not identical to the knowledge-disabled YAML pipeline.
