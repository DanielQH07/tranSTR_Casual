import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm


QUESTION_TYPES = (
    "counterfactual_reason",
    "predictive_reason",
    "counterfactual",
    "predictive",
    "explanatory",
    "descriptive",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the fixed TranSTR run1 pipeline from a YAML configuration."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print the configuration without loading data or a model",
    )
    return parser.parse_args()


def load_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML root must be a mapping")
    validate_config(cfg)
    return cfg, path


def validate_config(cfg):
    required_sections = {
        "run", "data", "model", "training", "resume", "wandb", "evaluation", "test"
    }
    missing = sorted(required_sections - set(cfg))
    if missing:
        raise KeyError(f"Missing YAML sections: {missing}")
    if cfg.get("schema_version") != 1:
        raise ValueError("Only schema_version=1 is supported")

    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    if data["source"] not in {"local", "kagglehub"}:
        raise ValueError("data.source must be 'local' or 'kagglehub'")
    if not data["return_family_id"]:
        raise ValueError("data.return_family_id must be true for NCOD LUM/HUM routing")
    if int(data["input_frames"]) < int(model["selected_frames"]):
        raise ValueError("model.selected_frames cannot exceed data.input_frames")
    if int(model["obj_feat_dim"]) <= int(model["obj_bbox_dim"]):
        raise ValueError("model.obj_feat_dim must be larger than model.obj_bbox_dim")
    if bool(model["obj_use_bbox_pos_embed"]) and int(model["d_model"]) % 8 != 0:
        raise ValueError("model.d_model must be divisible by 8 for bbox positional embedding")
    if training["model_optimizer"].lower() != "adamw":
        raise ValueError("run1 requires training.model_optimizer=adamw")
    if training["ncod"]["optimizer"].lower() != "sgd":
        raise ValueError("run1 requires training.ncod.optimizer=sgd")
    if training["scheduler"]["type"] != "cosine_warmup":
        raise ValueError("run1 requires training.scheduler.type=cosine_warmup")
    if int(training["batch_size"]) < 1 or int(training["accumulation_steps"]) < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    if int(data["num_workers"]) < 0:
        raise ValueError("data.num_workers cannot be negative")
    if cfg["wandb"]["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    weights = cfg["evaluation"]["weighted_score"]
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("evaluation.weighted_score values must sum to 1.0")


def print_config_summary(cfg, config_path):
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    print(f"Configuration OK: {config_path}")
    print(f"Run: {cfg['run']['name']} | seed={cfg['run']['seed']} | GPU={cfg['run']['gpu']}")
    print(
        f"Batch: {training['batch_size']} x accumulation {training['accumulation_steps']} "
        f"= {int(training['batch_size']) * int(training['accumulation_steps'])}"
    )
    print(
        f"Features: frames={data['input_frames']} -> selected={model['selected_frames']} | "
        f"objects={data['objects_per_frame']} | object_dim={model['obj_feat_dim']}"
    )
    print(
        f"Epochs={training['epochs']} | main_lr={training['main_lr']} | "
        f"text_lr={training['text_encoder_lr']} | workers={data['num_workers']}"
    )


def _expand_path(value, config_dir):
    if value is None:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = config_dir / expanded
    return expanded.resolve()


def _read_secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata

        return (userdata.get(name) or "").strip()
    except Exception:
        return ""


def set_reproducibility(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["run"]["gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["run"]["require_cuda"] and device.type != "cuda":
        raise RuntimeError("CUDA GPU is required by run.require_cuda")
    if device.type == "cuda":
        print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: cpu")
    return device


def setup_logger(output_dir, run_name, phase):
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"transtr.{phase}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / f"{phase}_{run_name}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    log.addHandler(stream)
    log.addHandler(file_handler)
    return log


def _download_kaggle_dataset(kagglehub, slug, override_env):
    override = os.environ.get(override_env, "").strip() if override_env else ""
    if override and Path(override).exists():
        return Path(override)
    return Path(kagglehub.dataset_download(slug))


def _find_dir_with_extension(root, extension):
    counts = {}
    for path in Path(root).rglob(f"*{extension}"):
        counts[path.parent] = counts.get(path.parent, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _find_named_directory(root, name):
    root = Path(root)
    if root.name.lower() == name.lower():
        return root
    return next(
        (path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()),
        None,
    )


def resolve_data_paths(cfg, config_dir):
    data_cfg = cfg["data"]
    if data_cfg["source"] == "local":
        paths = {
            key: _expand_path(data_cfg["local_paths"].get(key), config_dir)
            for key in ("video_feature_root", "grounding_dino_path", "sample_list_path", "split_dir")
        }
    else:
        kaggle_cfg = data_cfg["kagglehub"]
        token_name = kaggle_cfg["token_env"]
        token = _read_secret(token_name)
        if not token:
            raise RuntimeError(
                f"Missing {token_name}. Add it to the environment or Colab Secrets."
            )
        os.environ[token_name] = token
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError("Install kagglehub before using data.source=kagglehub") from exc

        overrides = kaggle_cfg["env_overrides"]
        gdino_slug = os.environ.get(
            overrides["grounding_dino_slug"], kaggle_cfg["grounding_dino_slug"]
        )
        dinov3_root = _download_kaggle_dataset(
            kagglehub, kaggle_cfg["dinov3_slug"], overrides["dinov3"]
        )
        gdino_root = _download_kaggle_dataset(
            kagglehub, gdino_slug, overrides["grounding_dino"]
        )
        annotation_root = _download_kaggle_dataset(
            kagglehub, kaggle_cfg["annotation_slug"], overrides["annotation"]
        )
        split_root = _download_kaggle_dataset(
            kagglehub, kaggle_cfg["split_slug"], overrides["split"]
        )

        source_features = list(dinov3_root.rglob("*.pt"))
        if not source_features:
            raise FileNotFoundError(f"No DINOv3 .pt files found under {dinov3_root}")
        merged = _expand_path(kaggle_cfg["merged_dinov3_path"], config_dir)
        merged.mkdir(parents=True, exist_ok=True)
        for source in source_features:
            destination = merged / source.name
            if destination.exists():
                continue
            try:
                destination.symlink_to(source)
            except Exception:
                shutil.copy2(source, destination)
        merged_count = len(list(merged.glob("*.pt")))
        unique_count = len({source.name for source in source_features})
        if merged_count != unique_count:
            raise RuntimeError(
                f"DINOv3 merge incomplete: merged={merged_count}, source_unique={unique_count}"
            )
        paths = {
            "video_feature_root": merged,
            "grounding_dino_path": _find_dir_with_extension(gdino_root, ".pkl") or gdino_root,
            "sample_list_path": _find_named_directory(annotation_root, "QA") or annotation_root,
            "split_dir": _find_named_directory(split_root, "split") or split_root,
        }

    for name, path in paths.items():
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Configured data path is missing: {name}={path}")
    if not list(Path(paths["video_feature_root"]).glob("*.pt")):
        raise RuntimeError(f"No .pt files in {paths['video_feature_root']}")
    if not list(Path(paths["grounding_dino_path"]).glob("*.pkl")):
        raise RuntimeError(f"No .pkl files in {paths['grounding_dino_path']}")
    return {key: str(value) for key, value in paths.items()}


class ObjectSafeCollator:
    def __init__(self, frames, objects, dimension, enabled=True):
        self.expected = (int(frames), int(objects), int(dimension))
        self.enabled = bool(enabled)

    def fit_sample(self, sample):
        if not self.enabled:
            return sample
        items = list(sample)
        obj = torch.as_tensor(items[1]).float()
        if obj.ndim != 3:
            raise RuntimeError(f"Expected object feature [T,O,D], got {tuple(obj.shape)}")
        fixed = obj.new_zeros(self.expected)
        sizes = tuple(min(obj.shape[index], self.expected[index]) for index in range(3))
        fixed[: sizes[0], : sizes[1], : sizes[2]] = obj[
            : sizes[0], : sizes[1], : sizes[2]
        ]
        items[1] = torch.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)
        return tuple(items)

    def __call__(self, batch):
        return default_collate([self.fit_sample(sample) for sample in batch])


def create_dataset(split, cfg, paths):
    from DataLoader import VideoQADataset

    data = cfg["data"]
    return VideoQADataset(
        split=split,
        n_query=int(data["answer_candidates"]),
        obj_num=int(data["objects_per_frame"]),
        sample_list_path=paths["sample_list_path"],
        video_feature_path=paths["video_feature_root"],
        grounding_dino_path=paths["grounding_dino_path"],
        split_dir=paths["split_dir"],
        topK_frame=int(data["input_frames"]),
        max_samples=data["max_samples"],
        verbose=True,
        return_family_id=bool(data["return_family_id"]),
    )


def create_collator(cfg):
    return ObjectSafeCollator(
        cfg["data"]["input_frames"],
        cfg["data"]["objects_per_frame"],
        cfg["model"]["obj_feat_dim"],
        cfg["data"]["object_feature_guard"],
    )


def create_loader(dataset, cfg, shuffle, collator):
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        collate_fn=collator,
    )


def unpack_batch(batch):
    if len(batch) == 7:
        return batch
    if len(batch) == 6:
        frame, obj, question, answers, answer_id, key = batch
        return frame, obj, question, answers, answer_id, key, None
    raise ValueError(f"Unexpected raw-text batch with {len(batch)} elements")


def split_question_key(question_key):
    key = str(question_key)
    for question_type in QUESTION_TYPES:
        suffix = f"_{question_type}"
        if key.endswith(suffix):
            return key[: -len(suffix)], question_type
    return key, "unknown"


def compute_acc_all_metrics(type_results):
    mapping = (
        ("Description", "descriptive"),
        ("Explanation", "explanatory"),
        ("Predictive-Answer", "predictive"),
        ("Predictive-Reason", "predictive_reason"),
        ("Counterfactual-Answer", "counterfactual"),
        ("Counterfactual-Reason", "counterfactual_reason"),
    )
    metrics = {}
    for metric_name, question_type in mapping:
        rows = type_results.get(question_type, [])
        metrics[metric_name] = (
            100.0 * sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
        )

    def paired_metric(answer_type, reason_type):
        answer_by_video = {
            row["video_id"]: row["correct"] for row in type_results.get(answer_type, [])
        }
        reason_by_video = {
            row["video_id"]: row["correct"] for row in type_results.get(reason_type, [])
        }
        common = set(answer_by_video) & set(reason_by_video)
        if not common:
            return 0.0
        correct = sum(
            1 for video_id in common if answer_by_video[video_id] and reason_by_video[video_id]
        )
        return 100.0 * correct / len(common)

    metrics["PAR"] = paired_metric("predictive", "predictive_reason")
    metrics["CAR"] = paired_metric("counterfactual", "counterfactual_reason")
    metrics["Acc_ALL"] = (
        metrics["Description"] + metrics["Explanation"] + metrics["PAR"] + metrics["CAR"]
    ) / 4.0
    return metrics


def model_kwargs_from_config(cfg, device):
    data = cfg["data"]
    model = cfg["model"]
    return {
        "text_encoder_type": model["text_encoder_type"],
        "freeze_text_encoder": bool(model["freeze_text_encoder"]),
        "n_query": int(data["answer_candidates"]),
        "objs": int(data["objects_per_frame"]),
        "frames": int(data["input_frames"]),
        "topK_frame": int(model["selected_frames"]),
        "topK_obj": int(model["topk_objects"]),
        "hard_eval": bool(model["hard_eval"]),
        "frame_feat_dim": int(model["frame_feat_dim"]),
        "obj_feat_dim": int(model["obj_feat_dim"]),
        "use_grounding_dino": bool(model["use_grounding_dino"]),
        "obj_use_bbox_pos_embed": bool(model["obj_use_bbox_pos_embed"]),
        "obj_hard_gather_from_frame": bool(model["obj_hard_gather_from_frame"]),
        "obj_bbox_dim": int(model["obj_bbox_dim"]),
        "obj_split_roi_class": bool(model["obj_split_roi_class"]),
        "obj_roi_dim": int(model["obj_roi_dim"]),
        "obj_class_dim": int(model["obj_class_dim"]),
        "obj_mask_padding": bool(model["obj_mask_padding"]),
        "d_model": int(model["d_model"]),
        "word_dim": int(model["word_dim"]),
        "nheads": int(model["nheads"]),
        "num_encoder_layers": int(model["num_encoder_layers"]),
        "normalize_before": bool(model["normalize_before"]),
        "activation": model["activation"],
        "dropout": float(model["dropout"]),
        "encoder_dropout": float(model["encoder_dropout"]),
        "num_question_families": int(model["num_question_families"]),
        "lambda_knowledge": float(model["lambda_knowledge"]),
        "device": device,
    }


def build_model(cfg, device):
    from networks.model import VideoQAmodel

    model = VideoQAmodel(**model_kwargs_from_config(cfg, device))
    if cfg["model"]["remove_knowledge_modules"]:
        for module_name in ("q_family_embed", "knowledge_head", "k_proj"):
            if hasattr(model, module_name):
                delattr(model, module_name)
    return model.to(device)


def build_optimizers(cfg, model, train_size, train_loader, device):
    training = cfg["training"]
    non_text = []
    text = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (text if "text_encoder" in name else non_text).append(parameter)
    groups = []
    if non_text:
        groups.append(
            {
                "params": non_text,
                "lr": float(training["main_lr"]),
                "weight_decay": float(training["weight_decay"]),
            }
        )
    if text:
        groups.append(
            {
                "params": text,
                "lr": float(training["text_encoder_lr"]),
                "weight_decay": float(training["weight_decay"]),
            }
        )
    if not groups:
        raise RuntimeError("No trainable model parameters")
    optimizer_model = torch.optim.AdamW(groups)

    accumulation = int(training["accumulation_steps"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_steps = max(1, int(training["epochs"]) * steps_per_epoch)
    warmup_steps = max(1, int(training["scheduler"]["warmup_epochs"]) * steps_per_epoch)
    main_lr = float(training["main_lr"])
    min_ratio = float(training["scheduler"]["min_lr"]) / max(main_lr, 1e-12)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_ratio, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lr_lambda)
    ncod = training["ncod"]
    uncertainty = nn.Parameter(
        torch.full((train_size,), float(ncod["u_init"]), dtype=torch.float32, device=device)
    )
    optimizer_u = torch.optim.SGD([uncertainty], lr=float(ncod["u_lr"]))
    return optimizer_model, optimizer_u, scheduler, uncertainty, text


def update_ema(ema_model, model, decay):
    if ema_model is None:
        return
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.data.mul_(decay).add_(parameter.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def hard_negative_weights(candidate_features, target, cfg):
    hard_negative = cfg["training"]["hard_negative"]
    maximum = float(hard_negative["max_weight"])
    if not hard_negative["enabled"] or candidate_features is None or maximum <= 1.0:
        return torch.ones_like(target, dtype=torch.float32)
    with torch.no_grad():
        normalized = F.normalize(candidate_features.detach(), dim=-1)
        gold = normalized.gather(
            1, target.view(-1, 1, 1).expand(-1, 1, normalized.size(-1))
        ).squeeze(1)
        similarities = torch.bmm(normalized, gold.unsqueeze(-1)).squeeze(-1)
        similarities.scatter_(1, target.view(-1, 1), -1.0)
        hardness = similarities.max(dim=1).values.clamp(min=0.0, max=1.0)
        return 1.0 + (maximum - 1.0) * hardness


def compute_sample_indices(question_keys, key_to_index, device):
    indices = [key_to_index.get(str(key), -1) for key in question_keys]
    if any(index < 0 for index in indices):
        missing = [str(question_keys[i]) for i, value in enumerate(indices) if value < 0][:5]
        raise KeyError(f"Missing train sample keys for NCOD U: {missing}")
    return torch.tensor(indices, dtype=torch.long, device=device)


def train_epoch(
    cfg,
    model,
    ema_model,
    loader,
    optimizer_model,
    optimizer_u,
    scheduler,
    uncertainty,
    key_to_index,
    device,
    epoch,
):
    model.train()
    training = cfg["training"]
    ncod = training["ncod"]
    accumulation = int(training["accumulation_steps"])
    verifier_weight = (
        0.0
        if epoch <= int(training["auxiliary_warmup_epochs"])
        else float(training["verifier_weight"])
    )
    bce = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "verifier": 0.0}
    correct = 0
    sample_count = 0
    optimizer_model.zero_grad(set_to_none=True)
    optimizer_u.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_index, batch in enumerate(progress):
        frame, obj, question, answers, answer_id, keys, family_id = unpack_batch(batch)
        frame = frame.to(device)
        obj = obj.to(device)
        target = answer_id.to(device)
        family_id = torch.zeros_like(target) if family_id is None else family_id.to(device)
        sample_indices = compute_sample_indices(keys, key_to_index, device)

        output = model(frame, obj, question, answers, return_aux=True, q_family_id=None)
        logits = output["logits"]
        verifier_logits = output.get("verifier_logits", logits)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.size(-1)).float()
        u_batch = uncertainty[sample_indices].unsqueeze(1)

        ce = -torch.sum(one_hot * torch.log(torch.clamp(probabilities, min=1e-12)), dim=1)
        shifted = torch.clamp(probabilities + u_batch.detach() * one_hot, min=1e-12, max=1.0)
        lum = -torch.sum(one_hot * torch.log(shifted), dim=1)
        hum = (1.0 + float(ncod["hum_alpha"]) * u_batch.detach().squeeze(1)) * ce
        hard_mask = family_id >= int(ncod["hard_family_min_id"])
        l1_per_sample = torch.where(hard_mask, hum, lum)
        l1 = (l1_per_sample * hard_negative_weights(output.get("cand_feat"), target, cfg)).mean()
        verifier_loss = bce(verifier_logits, one_hot)
        model_loss = l1 + verifier_weight * verifier_loss
        (model_loss / accumulation).backward()

        shifted_for_u = probabilities.detach() + u_batch * one_hot
        l2 = F.mse_loss(shifted_for_u, one_hot)
        (l2 / accumulation).backward()

        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(training["gradient_clip_max_norm"])
            )
            optimizer_model.step()
            update_ema(ema_model, model, float(training["ema"]["decay"]))
            scheduler.step()
            optimizer_model.zero_grad(set_to_none=True)
            optimizer_u.step()
            optimizer_u.zero_grad(set_to_none=True)
            with torch.no_grad():
                uncertainty.clamp_(float(ncod["u_clamp_min"]), float(ncod["u_clamp_max"]))

        totals["l1"] += l1.item()
        totals["l2"] += l2.item()
        totals["verifier"] += verifier_loss.item()
        totals["loss"] += (model_loss + l2).item()
        correct += (logits.argmax(dim=-1) == target).sum().item()
        sample_count += target.size(0)
        progress.set_postfix(loss=totals["loss"] / (batch_index + 1))

    batches = max(1, len(loader))
    return {
        "train_total_loss": totals["loss"] / batches,
        "train_l1": totals["l1"] / batches,
        "train_l2": totals["l2"] / batches,
        "train_verifier_loss": totals["verifier"] / batches,
        "train_acc": 100.0 * correct / max(1, sample_count),
        "lambda_verifier_eff": verifier_weight,
    }


def evaluate_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    type_results = {}
    with torch.inference_mode():
        for batch in loader:
            frame, obj, question, answers, answer_id, keys, _ = unpack_batch(batch)
            target = answer_id.to(device)
            output = model(
                frame.to(device), obj.to(device), question, answers, return_aux=True, q_family_id=None
            )
            predictions = output["logits"].argmax(dim=-1)
            correct += (predictions == target).sum().item()
            total += target.size(0)
            for key, prediction, expected in zip(
                keys, predictions.cpu().tolist(), target.cpu().tolist()
            ):
                video_id, question_type = split_question_key(key)
                type_results.setdefault(question_type, []).append(
                    {"video_id": video_id, "correct": int(prediction) == int(expected)}
                )
    metrics = compute_acc_all_metrics(type_results)
    metrics["Plain_Acc"] = 100.0 * correct / max(1, total)
    return metrics


def _build_evaluation_metadata(loader, candidate_count):
    sample_list = getattr(getattr(loader, "dataset", None), "sample_list", None)
    metadata = {}
    if sample_list is None:
        return metadata
    for _, row in sample_list.iterrows():
        video_id = str(row.get("video_id", ""))
        question_type = str(row.get("type", "unknown"))
        metadata[f"{video_id}_{question_type}"] = {
            "video_id": video_id,
            "question_type": question_type,
            "question": str(row.get("question", "")),
            "answers": [
                str(row.get(f"a{index}", "")) for index in range(candidate_count)
            ],
        }
    return metadata


def evaluate_detailed(cfg, model, loader, device, best_val_acc, csv_path):
    model.eval()
    candidate_count = int(cfg["data"]["answer_candidates"])
    metadata = _build_evaluation_metadata(loader, candidate_count)
    type_results = {}
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Test evaluation"):
            frame, obj, question, answers, answer_id, keys, _ = unpack_batch(batch)
            output = model(
                frame.to(device), obj.to(device), question, answers, return_aux=True, q_family_id=None
            )
            probabilities = torch.softmax(output["logits"], dim=-1).cpu().numpy()
            predictions = probabilities.argmax(axis=-1)
            targets = answer_id.numpy()
            for key, prediction, target, probability in zip(keys, predictions, targets, probabilities):
                video_id_fallback, question_type_fallback = split_question_key(key)
                meta = metadata.get(str(key), {})
                video_id = str(meta.get("video_id", video_id_fallback))
                question_type = str(meta.get("question_type", question_type_fallback))
                candidates = list(
                    meta.get("answers", [""] * candidate_count)
                )[:candidate_count]
                candidates += [""] * (candidate_count - len(candidates))
                prediction = int(prediction)
                target = int(target)
                is_correct = prediction == target
                type_results.setdefault(question_type, []).append(
                    {"video_id": video_id, "correct": is_correct}
                )
                row = {
                    "video_id": video_id,
                    "question_type": question_type,
                    "question": str(meta.get("question", "")),
                    "correct_idx": target,
                    "predicted_idx": prediction,
                    "is_correct": int(is_correct),
                    "confidence": float(probability[prediction]),
                    "predicted_answer": candidates[prediction],
                    "correct_answer": candidates[target],
                }
                for index in range(candidate_count):
                    row[f"a{index}"] = candidates[index]
                    row[f"prob_a{index}"] = float(probability[index])
                rows.append(row)
    if not rows:
        raise RuntimeError("Evaluation produced zero predictions")
    prediction_frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(csv_path, index=False)
    metrics = compute_acc_all_metrics(type_results)
    metrics["Plain_Acc"] = 100.0 * float(prediction_frame["is_correct"].mean())
    weights = cfg["evaluation"]["weighted_score"]
    metrics["WeightedScore_WeakPriority"] = (
        float(weights["predictive_reason"]) * metrics["Predictive-Reason"]
        + float(weights["counterfactual_reason"]) * metrics["Counterfactual-Reason"]
        + float(weights["acc_all"]) * metrics["Acc_ALL"]
        + float(weights["best_val_acc"]) * float(best_val_acc)
    )
    return metrics, type_results


def build_output_paths(cfg, config_dir):
    output_dir = _expand_path(cfg["run"]["output_dir"], config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["run"]["name"]
    return {
        "dir": output_dir,
        "best": output_dir / f"best_model_{run_name}.ckpt",
        "last": output_dir / f"last_checkpoint_{run_name}.ckpt",
        "history": output_dir / f"train_history_{run_name}.csv",
        "predictions": output_dir / f"test_predictions_{run_name}.csv",
        "metrics": output_dir / f"final_metrics_{run_name}.json",
        "comparison": output_dir / "run1_comparison.csv",
    }


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_state(model, state, strict=True):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model state must be a mapping")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    current = model.state_dict()
    shape_errors = []
    filtered = {}
    unexpected = []
    for key, value in state.items():
        if key not in current:
            unexpected.append(key)
        elif current[key].shape != value.shape:
            shape_errors.append(f"{key}: checkpoint={tuple(value.shape)} model={tuple(current[key].shape)}")
        else:
            filtered[key] = value
    ignored_prefixes = ("q_family_embed.", "knowledge_head.", "k_proj.")
    missing = [key for key in current if key not in filtered]
    critical_missing = [key for key in missing if not key.startswith(ignored_prefixes)]
    critical_unexpected = [
        key for key in unexpected if not key.startswith(ignored_prefixes)
    ]
    if strict and (shape_errors or critical_missing or critical_unexpected):
        raise RuntimeError(
            "Incompatible checkpoint:\n"
            + "\n".join(shape_errors[:20])
            + f"\nmissing={critical_missing[:20]}\nunexpected={critical_unexpected[:20]}"
        )
    result = model.load_state_dict(filtered, strict=False)
    return result.missing_keys, unexpected


def init_wandb(cfg, model):
    wandb_cfg = cfg["wandb"]
    mode = wandb_cfg["mode"]
    if mode == "disabled":
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or set wandb.mode=disabled") from exc
    if mode == "online":
        key_name = wandb_cfg["api_key_env"]
        api_key = _read_secret(key_name)
        if not api_key:
            raise RuntimeError(f"Missing {key_name}; use W&B offline/disabled or provide the key")
        os.environ[key_name] = api_key
        os.environ.pop("WANDB_MODE", None)
        wandb.login(key=api_key, verify=True)
    else:
        os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=cfg["run"]["name"],
        config=cfg,
        reinit=True,
    )
    if wandb_cfg["watch_gradients"]:
        wandb.watch(
            model,
            log="gradients",
            log_freq=int(wandb_cfg["gradient_log_frequency"]),
        )
    return wandb, run


def resolve_resume_checkpoint(cfg, config_dir, paths, wandb_module):
    resume = cfg["resume"]
    if not resume["enabled"]:
        return None
    source = resume["source"]
    if source == "auto":
        source = "wandb" if cfg["wandb"]["mode"] == "online" else "local"
    if source == "local":
        configured = resume["local_checkpoint_path"]
        return _expand_path(configured, config_dir) if configured else paths["last"]
    if source != "wandb":
        raise ValueError("resume.source must be auto, local, or wandb")
    if wandb_module is None:
        raise RuntimeError("W&B resume requested while W&B is disabled")
    api = wandb_module.Api()
    entity = cfg["wandb"]["entity"] or api.default_entity
    artifact_name = f"last-checkpoint-{cfg['run']['name']}"
    artifact_path = (
        f"{entity}/{cfg['wandb']['project']}/{artifact_name}:{resume['artifact_alias']}"
    )
    artifact_dir = Path(api.artifact(artifact_path).download())
    expected = artifact_dir / paths["last"].name
    if expected.exists():
        return expected
    candidates = list(artifact_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in W&B artifact {artifact_path}")
    return candidates[0]


def restore_training_state(
    checkpoint,
    model,
    ema_model,
    optimizer_model,
    optimizer_u,
    scheduler,
    uncertainty,
    device,
):
    load_model_state(model, checkpoint["model_state_dict"], strict=True)
    optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])
    optimizer_u.load_state_dict(checkpoint["optimizer_u_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if ema_model is not None and checkpoint.get("ema_model_state_dict") is not None:
        load_model_state(ema_model, checkpoint["ema_model_state_dict"], strict=True)
    saved_u = checkpoint.get("U")
    if saved_u is not None:
        saved_u = saved_u.to(device).float().view(-1)
        with torch.no_grad():
            count = min(saved_u.numel(), uncertainty.numel())
            uncertainty[:count].copy_(saved_u[:count])
    return {
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "best_acc": float(checkpoint.get("best_acc", float("-inf"))),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
        "history_rows": list(checkpoint.get("history_rows", [])),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def log_finish_artifacts(cfg, paths, wandb_module, run):
    if wandb_module is None or run is None:
        return
    run_name = cfg["run"]["name"]
    if cfg["wandb"]["upload_checkpoints_at_finish"]:
        if paths["last"].exists():
            artifact = wandb_module.Artifact(
                f"last-checkpoint-{run_name}", type="model", metadata={"kind": "last"}
            )
            artifact.add_file(str(paths["last"]), name=paths["last"].name)
            wandb_module.log_artifact(artifact, aliases=["last", "latest"])
        if paths["best"].exists():
            artifact = wandb_module.Artifact(
                f"best-model-{run_name}", type="model", metadata={"kind": "best"}
            )
            artifact.add_file(str(paths["best"]), name=paths["best"].name)
            wandb_module.log_artifact(artifact, aliases=["best"])
    results = wandb_module.Artifact(f"final-results-{run_name}", type="results")
    for key in ("metrics", "predictions", "history", "comparison"):
        if paths[key].exists():
            results.add_file(str(paths[key]), name=paths[key].name)
    wandb_module.log_artifact(results)


def main():
    cli = parse_args()
    cfg, config_path = load_config(cli.config)
    print_config_summary(cfg, config_path)
    if cli.check_config:
        return

    config_dir = config_path.parent
    paths = build_output_paths(cfg, config_dir)
    log = setup_logger(paths["dir"], cfg["run"]["name"], "train")
    device = resolve_device(cfg)
    set_reproducibility(cfg["run"]["seed"])
    data_paths = resolve_data_paths(cfg, config_dir)

    train_dataset = create_dataset("train", cfg, data_paths)
    val_dataset = create_dataset("val", cfg, data_paths)
    test_dataset = create_dataset("test", cfg, data_paths)
    for name, dataset in (
        ("train", train_dataset), ("val", val_dataset), ("test", test_dataset)
    ):
        if len(dataset) == 0:
            raise RuntimeError(f"{name} dataset is empty")
    collator = create_collator(cfg)
    raw_probe = train_dataset[0]
    fixed_probe = collator.fit_sample(raw_probe)
    log.info("Object feature guard: %s -> %s", tuple(raw_probe[1].shape), tuple(fixed_probe[1].shape))
    train_loader = create_loader(train_dataset, cfg, True, collator)
    val_loader = create_loader(val_dataset, cfg, False, collator)
    test_loader = create_loader(test_dataset, cfg, False, collator)

    train_keys = [
        f"{row.video_id}_{row.type}" for row in train_dataset.sample_list.itertuples(index=False)
    ]
    key_to_index = {key: index for index, key in enumerate(train_keys)}
    model = build_model(cfg, device)
    ema_model = None
    if cfg["training"]["ema"]["enabled"]:
        ema_model = copy.deepcopy(model).to(device).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    optimizer_model, optimizer_u, scheduler, uncertainty, text_parameters = build_optimizers(
        cfg, model, len(train_dataset), train_loader, device
    )
    log.info(
        "Trainable parameters: %.2fM | text encoder: %.2fM",
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6,
        sum(parameter.numel() for parameter in text_parameters) / 1e6,
    )
    wandb_module, wandb_run = init_wandb(cfg, model)

    state = {
        "start_epoch": 1,
        "best_acc": float("-inf"),
        "best_epoch": 0,
        "epochs_without_improvement": 0,
        "history_rows": [],
    }
    if cfg["resume"]["enabled"]:
        try:
            resume_path = resolve_resume_checkpoint(
                cfg, config_dir, paths, wandb_module
            )
            if resume_path is None or not Path(resume_path).exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            checkpoint = load_torch_checkpoint(resume_path, device)
            state = restore_training_state(
                checkpoint,
                model,
                ema_model,
                optimizer_model,
                optimizer_u,
                scheduler,
                uncertainty,
                device,
            )
            log.info("Resumed from %s at epoch %d", resume_path, state["start_epoch"])
        except Exception as exc:
            log.warning("Resume failed; training from epoch 1: %s", exc)

    early = cfg["training"]["early_stopping"]
    for epoch in range(state["start_epoch"], int(cfg["training"]["epochs"]) + 1):
        train_metrics = train_epoch(
            cfg,
            model,
            ema_model,
            train_loader,
            optimizer_model,
            optimizer_u,
            scheduler,
            uncertainty,
            key_to_index,
            device,
            epoch,
        )
        evaluation_model = ema_model if ema_model is not None else model
        val_metrics = evaluate_epoch(evaluation_model, val_loader, device)
        val_acc = float(val_metrics["Acc_ALL"])
        improved = val_acc > state["best_acc"] + float(early["min_delta"])
        if improved:
            state["best_acc"] = val_acc
            state["best_epoch"] = epoch
            state["epochs_without_improvement"] = 0
        elif epoch >= int(early["start_epoch"]):
            state["epochs_without_improvement"] += 1

        learning_rates = [group["lr"] for group in optimizer_model.param_groups]
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_acc": val_acc,
            "val_plain_acc": float(val_metrics["Plain_Acc"]),
            "val_Description": float(val_metrics["Description"]),
            "val_Explanation": float(val_metrics["Explanation"]),
            "val_PAR": float(val_metrics["PAR"]),
            "val_CAR": float(val_metrics["CAR"]),
            "u_mean": float(uncertainty.detach().mean().item()),
            "u_max": float(uncertainty.detach().max().item()),
            "lr_min": float(min(learning_rates)),
            "lr_max": float(max(learning_rates)),
            "best_acc_so_far": float(state["best_acc"]),
            "best_epoch_so_far": int(state["best_epoch"]),
            "epochs_without_improvement": int(state["epochs_without_improvement"]),
        }
        state["history_rows"].append(row)
        pd.DataFrame(state["history_rows"]).to_csv(paths["history"], index=False)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_model_state_dict": optimizer_model.state_dict(),
            "optimizer_u_state_dict": optimizer_u.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": state["best_acc"],
            "best_epoch": state["best_epoch"],
            "epochs_without_improvement": state["epochs_without_improvement"],
            "history_rows": state["history_rows"],
            "U": uncertainty.detach().cpu(),
            "train_sample_keys": train_keys,
            "config": cfg,
        }
        torch.save(checkpoint, paths["last"])
        if improved:
            torch.save(checkpoint, paths["best"])
        if wandb_module is not None:
            wandb_module.log(row)
        log.info(
            "Epoch %d | train_acc=%.2f | val_Acc_ALL=%.2f | best=%.2f (epoch %d)",
            epoch,
            row["train_acc"],
            val_acc,
            state["best_acc"],
            state["best_epoch"],
        )
        if (
            epoch >= int(early["start_epoch"])
            and state["epochs_without_improvement"] >= int(early["patience"])
        ):
            log.info("Early stopping at epoch %d", epoch)
            break

    if not paths["best"].exists():
        raise RuntimeError("Training ended without a best checkpoint")
    best_checkpoint = load_torch_checkpoint(paths["best"], device)
    best_state = best_checkpoint.get("ema_model_state_dict") or best_checkpoint["model_state_dict"]
    load_model_state(model, best_state, strict=True)
    metrics, _ = evaluate_detailed(
        cfg, model, test_loader, device, state["best_acc"], paths["predictions"]
    )
    write_json(paths["metrics"], metrics)
    comparison_row = {
        "run_tag": cfg["run"]["name"],
        "best_val_acc": float(state["best_acc"]),
        "best_epoch": int(state["best_epoch"]),
        **{key: float(value) for key, value in metrics.items()},
    }
    if paths["comparison"].exists():
        comparison = pd.read_csv(paths["comparison"])
        comparison = comparison[comparison["run_tag"] != cfg["run"]["name"]]
        comparison = pd.concat([comparison, pd.DataFrame([comparison_row])], ignore_index=True)
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison.to_csv(paths["comparison"], index=False)

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_val_acc": float(state["best_acc"]),
                "best_epoch": int(state["best_epoch"]),
                "weighted_score_weak_priority": float(metrics["WeightedScore_WeakPriority"]),
            }
        )
        wandb_module.log(
            {f"eval/{key.replace('-', '_')}": float(value) for key, value in metrics.items()}
        )
    log_finish_artifacts(cfg, paths, wandb_module, wandb_run)
    if wandb_module is not None:
        wandb_module.finish()
    log.info("Finished. PAR=%.2f | CAR=%.2f | Acc_ALL=%.2f", metrics["PAR"], metrics["CAR"], metrics["Acc_ALL"])


if __name__ == "__main__":
    main()
