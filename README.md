# TRUST — Causal VideoQA

**TRUST: Task-Routed Uncertainty-aware Spatio-Temporal Reasoning for Causal VideoQA**

TRUST kết hợp định tuyến theo loại tác vụ/câu hỏi, học nhận biết bất định bằng NCOD/LUM/HUM,
suy luận không-thời gian từ DINOv3 và GroundingDINO/Faster R-CNN, verifier loss,
hard-negative weighting và EMA.

TRUST combines task/question routing, uncertainty-aware NCOD/LUM/HUM learning,
spatio-temporal DINOv3 and GroundingDINO/Faster R-CNN reasoning, verifier loss,
hard-negative weighting, and EMA.

**Nguồn cấu hình duy nhất / single configuration source:** [`config.yaml`](./config.yaml).
`train.py` và `test.py` chỉ nhận đường dẫn YAML; không chỉnh hyperparameter trực tiếp trong script.

## Cấu trúc mã nguồn / Source structure

Cấu trúc dưới đây được đối chiếu với toàn bộ file Git đang theo dõi hoặc chưa bị ignore.
`backup/` và các file/thư mục khớp `.gitignore` không thuộc gói mã nguồn bên dưới.

```text
causalvid/
├── .gitignore
├── README.md
├── HuongDanCaiDat.txt
├── HuongDanSuDung.txt
├── requirements.txt
├── config.yaml
├── train.py
├── test.py
├── eval_mc.py
├── DataLoader.py
├── networks/
│   ├── model.py
│   ├── attention.py
│   ├── multimodal_transformer.py
│   ├── position_encoding.py
│   ├── topk.py
│   ├── question_router.py
│   ├── knowledge_retriever.py
│   ├── encoder.py
│   ├── EncoderVid.py
│   └── util.py
├── utils/
│   ├── util.py
│   └── logger.py
├── tools/
│   ├── download_weight.py
│   ├── extractvit.ipynb
│   └── Groundingdino-FasterRCNN-feat.ipynb
└── example/
    ├── train_colab.ipynb
    ├── train_kaggle.ipynb
    ├── inference_colab.ipynb
    └── inference_kaggle.ipynb
```

| Thành phần | Vai trò / Purpose |
|---|---|
| `config.yaml` | Cấu hình chung duy nhất cho train/test |
| `train.py` | Huấn luyện TRUST, validation, best/last checkpoint và metrics |
| `test.py` | Load checkpoint, đánh giá và xuất prediction CSV |
| `eval_mc.py` | Tiện ích đánh giá multiple-choice |
| `DataLoader.py` | Đọc DINOv3, object feature, QA và split |
| `networks/` | TRUST model, attention, transformer, Top-K và task routing |
| `utils/` | Seed/GPU/path/logging utilities |
| `tools/download_weight.py` | Tải checkpoint dựng sẵn vào `weights/` |
| `tools/extractvit.ipynb` | Sinh DINOv3 `[16,1024]` từ raw MP4 |
| `tools/Groundingdino-FasterRCNN-feat.ipynb` | Sinh object feature 2820 chiều |
| `example/train_colab.ipynb` | Luồng train Colab ưu tiên, batch 32/workers 0 |
| `example/train_kaggle.ipynb` | Luồng train Kaggle, biến thể n2 legacy |
| `example/inference_colab.ipynb` | Full test inference trên Colab |
| `example/inference_kaggle.ipynb` | Full test inference trên Kaggle và đối chiếu raw MP4 |
| `HuongDanCaiDat.txt` | Hướng dẫn môi trường, thư viện và credential |
| `HuongDanSuDung.txt` | Hướng dẫn chạy feature/train/test/inference |

> Tên repository/folder `tranSTR_Casual`, một số đường dẫn clone, checkpoint và tên notebook nguồn
> được giữ nguyên để tương thích với tài nguyên đã công bố. Đây là định danh legacy, không phải tên phương pháp hiện tại.
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

`DataLoader.py` yêu cầu DINOv3 `.pt` nằm phẳng trong `data/visual_feature/`. Object `.pkl` có thể
nằm phẳng hoặc trong các thư mục con trực tiếp mà DataLoader quét được.

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
python tools/download_weight.py
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
python tools/download_weight.py
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
resolved MP4 path to the output CSV. TRUST inference still consumes precomputed DINOv3/GDINO features.

> `example/train_kaggle.ipynb` intentionally retains the original n2 knowledge loss and memory
> post-check; it is not identical to the knowledge-disabled YAML pipeline.
