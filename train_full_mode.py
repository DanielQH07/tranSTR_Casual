#!/usr/bin/env python3
"""Command-line local runner converted from Train_full_mode.ipynb.

This script keeps the notebook's full-mode training path, but replaces Kaggle/
Colab setup with local input paths and Linux-friendly logging.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

np = None
pd = None
torch = None
nn = None
F = None
ReduceLROnPlateau = None
DataLoader = None
tqdm = None
VideoQADataset = None
VideoQAmodel = None
set_seed = None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(run_name: str) -> Path:
    log_dir = Path.cwd() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("=" * 90)
    print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Log file: {log_path}")
    print("=" * 90)
    return log_path


def require_path(label: str, path: str, must_contain_ext: Optional[str] = None) -> None:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    if must_contain_ext and p.is_dir():
        matches = list(p.glob(f"*{must_contain_ext}"))
        if not matches:
            raise FileNotFoundError(f"{label} has no {must_contain_ext} files: {p}")
    preview = list(p.iterdir())[:3] if p.is_dir() else []
    preview_names = [x.name for x in preview]
    print(f"OK {label}: {p}" + (f" | preview={preview_names}" if preview_names else ""))


def load_env_file(env_file: str) -> None:
    env_path = Path(env_file).expanduser()
    if not env_path.exists():
        print(f".env not found, skipping: {env_path}")
        return

    loaded_keys = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded_keys.append(key)

    if "WANDB_TOKEN" in os.environ and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
        loaded_keys.append("WANDB_API_KEY(from WANDB_TOKEN)")

    if loaded_keys:
        printable = [k for k in loaded_keys if "KEY" not in k and "TOKEN" not in k]
        hidden = len(loaded_keys) - len(printable)
        suffix = f" + {hidden} secret(s)" if hidden else ""
        print(f"Loaded .env keys: {printable}{suffix}")
    else:
        print(f".env loaded but no new keys were added: {env_path}")


def nvidia_gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def ensure_torch_gpu_if_needed(args) -> None:
    if args.allow_cpu or not args.auto_install_torch_gpu:
        return

    try:
        current_torch = importlib.import_module("torch")
        torch_import_error = None
    except Exception as exc:
        current_torch = None
        torch_import_error = exc

    if current_torch is not None and current_torch.cuda.is_available():
        return

    if not nvidia_gpu_available():
        return

    if os.environ.get("TRAIN_FULL_MODE_TORCH_GPU_INSTALL_ATTEMPTED") == "1":
        print("Torch GPU auto-install was already attempted once; continuing to environment check.")
        return

    if current_torch is None:
        print(f"Torch import failed while NVIDIA GPU is visible: {torch_import_error}")
    else:
        print(
            "Torch is installed but CUDA is unavailable "
            f"(torch={current_torch.__version__}, torch.version.cuda={current_torch.version.cuda})."
        )

    packages = args.torch_gpu_packages.split()
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        *packages,
        "--index-url",
        args.torch_cuda_index_url,
    ]
    print("Installing PyTorch GPU build:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

    os.environ["TRAIN_FULL_MODE_TORCH_GPU_INSTALL_ATTEMPTED"] = "1"
    print("PyTorch GPU build installed. Restarting script to load the new torch package...")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def gpu_status(prefix: str = "GPU") -> str:
    if not torch.cuda.is_available():
        return f"{prefix}: cuda unavailable"
    rows = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        total = props.total_memory / 1024**3
        rows.append(
            f"cuda:{idx} {props.name} | alloc={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB total={total:.2f}GB"
        )
    return f"{prefix}: " + " ; ".join(rows)


def check_environment(require_gpu: bool = True) -> None:
    global torch
    print("=== Environment check ===")
    modules = [
        "torch",
        "numpy",
        "pandas",
        "tqdm",
        "transformers",
        "sentencepiece",
        "einops",
        "wandb",
    ]
    missing = []
    for module in modules:
        try:
            imported = importlib.import_module(module)
            if module == "torch":
                torch = imported
            print(f"OK import {module}")
        except Exception as exc:
            missing.append(f"{module}: {exc}")
            print(f"MISSING import {module}: {exc}")

    print(f"Python: {sys.version.split()[0]}")
    if torch is None:
        print("PyTorch: unavailable")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(gpu_status("GPU before train"))

    if torch is not None and torch.cuda.is_available():
        try:
            x = torch.ones((8, 8), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print(f"OK cuda tensor test: mean={y.mean().item():.2f}")
            del x, y
            torch.cuda.empty_cache()
        except Exception as exc:
            missing.append(f"pytorch-gpu tensor test: {exc}")
            print(f"FAILED cuda tensor test: {exc}")
    elif require_gpu and torch is not None:
        missing.append("pytorch-gpu: torch.cuda.is_available() is False")

    if missing:
        raise RuntimeError(
            "Environment check failed. Install/fix these before training:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    print("Environment check passed.")


def import_runtime_modules():
    global np, pd, torch, nn, F, ReduceLROnPlateau, DataLoader, tqdm

    import numpy as _np
    import pandas as _pd
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm.auto import tqdm as _tqdm

    np = _np
    pd = _pd
    torch = _torch
    nn = _nn
    F = _F
    ReduceLROnPlateau = _ReduceLROnPlateau
    DataLoader = _DataLoader
    tqdm = _tqdm


def import_project_modules():
    global VideoQADataset, VideoQAmodel, set_seed
    from DataLoader import VideoQADataset as _VideoQADataset
    from networks.model import VideoQAmodel as _VideoQAmodel
    from utils.util import set_seed as _set_seed

    VideoQADataset = _VideoQADataset
    VideoQAmodel = _VideoQAmodel
    set_seed = _set_seed


def _unpack_batch(batch):
    if len(batch) == 7:
        ff, of, q, a, ans_id, qns_key, q_family_id = batch
    elif len(batch) == 6:
        ff, of, q, a, ans_id, qns_key = batch
        q_family_id = None
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements")
    return ff, of, q, a, ans_id, qns_key, q_family_id


def _compute_sample_indices(qns_keys, key_to_idx, device):
    idxs = [key_to_idx.get(str(k), -1) for k in qns_keys]
    if any(i < 0 for i in idxs):
        missing = [str(qns_keys[i]) for i, v in enumerate(idxs) if v < 0][:5]
        raise KeyError(f"Missing qns_key in key_to_idx mapping: {missing}")
    return torch.tensor(idxs, dtype=torch.long, device=device)


def train_epoch_integrated(
    model,
    optimizer_model,
    optimizer_u,
    U,
    loader,
    bce,
    device,
    epoch,
    key_to_idx,
    accumulation_steps=4,
    hum_alpha=1.0,
    lambda_verifier=0.2,
    lambda_knowledge=0.1,
):
    model.train()
    total_loss, total_l1, total_l2 = 0.0, 0.0, 0.0
    total_verifier, total_knowledge = 0.0, 0.0
    correct, total = 0, 0
    optimizer_model.zero_grad()
    optimizer_u.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        ff, of, q, a, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
        ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)

        if q_family_id is None:
            q_family_id = torch.zeros_like(tgt)
        else:
            q_family_id = q_family_id.to(device)

        sample_indices = _compute_sample_indices(qns_keys, key_to_idx, device)
        out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
        logits = out["logits"]
        fused_logits = out.get("fused_score", logits)
        verifier_logits = out.get("verifier_logits", logits)
        knowledge_logits = out.get("knowledge_score", None)

        probs = torch.softmax(logits, dim=1)
        y_onehot = F.one_hot(tgt, num_classes=logits.size(-1)).float()
        u_batch = U[sample_indices].unsqueeze(1)

        ce_per_sample = -torch.sum(y_onehot * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        shifted_probs = torch.clamp(probs + (u_batch.detach() * y_onehot), min=1e-12, max=1.0)
        lum_loss = -torch.sum(y_onehot * torch.log(shifted_probs), dim=1)
        hum_loss = (1.0 + hum_alpha * u_batch.detach().squeeze(1)) * ce_per_sample

        hard_mask = q_family_id >= 2
        l1 = torch.where(hard_mask, hum_loss, lum_loss).mean()

        verifier_loss = bce(verifier_logits, y_onehot)
        if knowledge_logits is not None:
            knowledge_loss = bce(knowledge_logits, y_onehot)
        else:
            knowledge_loss = torch.tensor(0.0, device=device)

        model_loss = l1 + lambda_verifier * verifier_loss + lambda_knowledge * knowledge_loss
        (model_loss / accumulation_steps).backward()

        shifted_det = probs.detach() + (u_batch * y_onehot)
        l2 = F.mse_loss(shifted_det, y_onehot)
        (l2 / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_model.step()
            optimizer_model.zero_grad()
            optimizer_u.step()
            optimizer_u.zero_grad()
            with torch.no_grad():
                U.clamp_(0.0, 0.99)

        total_l1 += l1.item()
        total_l2 += l2.item()
        total_verifier += verifier_loss.item()
        total_knowledge += knowledge_loss.item()
        total_loss += (model_loss + l2).item()
        correct += (fused_logits.argmax(-1) == tgt).sum().item()
        total += tgt.size(0)

        pbar.set_postfix(
            {
                "loss": total_loss / (batch_idx + 1),
                "l1": total_l1 / (batch_idx + 1),
                "l2": total_l2 / (batch_idx + 1),
                "ver": total_verifier / (batch_idx + 1),
                "know": total_knowledge / (batch_idx + 1),
                "acc": correct / max(total, 1) * 100,
            }
        )

    n = len(loader)
    return (
        total_loss / n,
        total_l1 / n,
        total_l2 / n,
        total_verifier / n,
        total_knowledge / n,
        correct / max(total, 1) * 100,
    )


def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ff, of, q, a, ans_id, _, q_family_id = _unpack_batch(batch)
            ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)
            q_family_id = q_family_id.to(device) if q_family_id is not None else None
            out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
            logits = out.get("fused_score", out["logits"])
            correct += (logits.argmax(-1) == tgt).sum().item()
            total += tgt.size(0)
    return correct / max(total, 1) * 100


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train full mode locally from Train_full_mode.ipynb")

    parser.add_argument("--run_name", required=True, help="Unique run name, used for W&B name, files, and log.")
    parser.add_argument("--run_profile", default="run1", choices=["baseline", "run1", "run2", "run3"])
    parser.add_argument("--run3_reg_mode", default="dropout", choices=["dropout", "decay"])

    parser.add_argument("--dinov3_feature_path", required=True, help="Local folder with merged DINOv3 .pt files.")
    parser.add_argument("--gdino_feature_path", required=True, help="Local folder with merged GDINO/FRCNN .pkl files.")
    parser.add_argument("--annotation_path", required=True, help="Local QA annotation folder.")
    parser.add_argument("--split_dir", required=True, help="Local split folder containing train.pkl, valid.pkl, test.pkl.")
    parser.add_argument("--model_dir", default=None, help="Default: ./models/<run_name>")

    parser.add_argument("--wandb_project", default="transtr-causalvid-dino_local")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--no_wandb_watch", action="store_true")
    parser.add_argument("--env_file", default=".env", help="Load WANDB_API_KEY/WANDB_TOKEN and other env vars from this file.")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--no_auto_install_torch_gpu", dest="auto_install_torch_gpu", action="store_false")
    parser.set_defaults(auto_install_torch_gpu=True)
    parser.add_argument(
        "--torch_cuda_index_url",
        default="https://download.pytorch.org/whl/cu121",
        help="PyTorch CUDA wheel index used when installed torch is CPU-only.",
    )
    parser.add_argument(
        "--torch_gpu_packages",
        default="torch torchvision torchaudio",
        help="Space-separated packages to install from --torch_cuda_index_url.",
    )
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.05)
    parser.add_argument("--early_stop_start_epoch", type=int, default=None)

    parser.add_argument("--lambda_verifier", type=float, default=None)
    parser.add_argument("--lambda_knowledge", type=float, default=None)
    parser.add_argument("--ncod_u_lr", type=float, default=0.1)
    parser.add_argument("--hum_alpha", type=float, default=1.0)

    parser.add_argument("--frame_feat_dim", type=int, default=1024)
    parser.add_argument("--obj_feat_dim", type=int, default=2820)
    parser.add_argument("--objs", type=int, default=12)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--topK_frame", type=int, default=16)
    parser.add_argument("--select_frames", type=int, default=5)
    parser.add_argument("--topK_obj", type=int, default=12)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--word_dim", type=int, default=768)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "glu"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--encoder_dropout", type=float, default=None)
    parser.set_defaults(normalize_before=True)
    parser.add_argument("--normalize_before", dest="normalize_before", action="store_true")
    parser.add_argument("--no_normalize_before", dest="normalize_before", action="store_false")

    parser.add_argument("--text_encoder_type", default="microsoft/deberta-base")
    parser.add_argument("--freeze_text_encoder", action="store_true")
    parser.add_argument("--text_pool_mode", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default="query_proj,key_proj,value_proj")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", default=None, help="Local checkpoint path. Defaults to latest checkpoint in model_dir.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")

    return parser


RUN_PROFILES = {
    "baseline": {
        "epoch": 10,
        "lr": 1e-5,
        "lambda_verifier": 0.3,
        "lambda_knowledge": 0.2,
        "early_stop_start_epoch": 5,
        "early_stop_patience": 4,
        "dropout": 0.3,
        "encoder_dropout": 0.3,
        "decay": 1e-4,
    },
    "run1": {
        "epoch": 10,
        "lr": 1e-5,
        "lambda_verifier": 0.25,
        "lambda_knowledge": 0.3,
        "early_stop_start_epoch": 5,
        "early_stop_patience": 4,
        "dropout": 0.3,
        "encoder_dropout": 0.3,
        "decay": 1e-4,
    },
    "run2": {
        "epoch": 10,
        "lr": 8e-6,
        "lambda_verifier": 0.25,
        "lambda_knowledge": 0.3,
        "early_stop_start_epoch": 6,
        "early_stop_patience": 5,
        "dropout": 0.3,
        "encoder_dropout": 0.3,
        "decay": 1e-4,
    },
    "run3": {
        "epoch": 10,
        "lr": 8e-6,
        "lambda_verifier": 0.25,
        "lambda_knowledge": 0.3,
        "early_stop_start_epoch": 6,
        "early_stop_patience": 5,
        "dropout": 0.3,
        "encoder_dropout": 0.3,
        "decay": 1e-4,
    },
}


def apply_profile(cli_args):
    profile = RUN_PROFILES[cli_args.run_profile].copy()
    if cli_args.run_profile == "run3":
        if cli_args.run3_reg_mode == "dropout":
            profile["dropout"] = 0.25
            profile["encoder_dropout"] = 0.25
        elif cli_args.run3_reg_mode == "decay":
            profile["decay"] = 8e-5

    for key in [
        "epoch",
        "lr",
        "lambda_verifier",
        "lambda_knowledge",
        "early_stop_start_epoch",
        "early_stop_patience",
        "dropout",
        "encoder_dropout",
        "decay",
    ]:
        if getattr(cli_args, key) is None:
            setattr(cli_args, key, profile[key])

    cli_args.lora_target_modules = [
        item.strip() for item in cli_args.lora_target_modules.split(",") if item.strip()
    ]
    if cli_args.use_lora and cli_args.freeze_text_encoder:
        raise ValueError("--freeze_text_encoder cannot be used with --use_lora")
    return cli_args


def build_model_config(args, device):
    cfg = vars(args).copy()
    cfg.update(
        {
            "video_feature_root": args.dinov3_feature_path,
            "grounding_dino_path": args.gdino_feature_path,
            "sample_list_path": args.annotation_path,
            "split_dir_txt": args.split_dir,
            "use_grounding_dino": True,
            "return_family_id": True,
            "hard_eval": False,
            "pos_ratio": 1.0,
            "neg_ratio": 1.0,
            "a": 1.0,
            "n_query": 5,
            "device": device,
            "topK_frame": args.select_frames,
        }
    )
    return SimpleNamespace(**cfg)


def create_dataloaders(args):
    print("=== Datasets ===")
    common = dict(
        n_query=args.n_query,
        obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_root,
        grounding_dino_path=args.grounding_dino_path,
        split_dir=args.split_dir_txt,
        topK_frame=args.topK_frame,
        max_samples=args.max_samples,
        verbose=True,
        return_family_id=args.return_family_id,
    )
    train_ds = VideoQADataset(split="train", **common)
    val_ds = VideoQADataset(split="val", **common)
    test_ds = VideoQADataset(split="test", **common)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    train_sample_keys = [f"{row.video_id}_{row.type}" for row in train_ds.sample_list.itertuples(index=False)]
    train_key_to_idx = {k: i for i, k in enumerate(train_sample_keys)}
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, train_sample_keys, train_key_to_idx


def init_wandb(args, model, wandb_config):
    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    import wandb

    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
    if wandb_key and args.wandb_mode != "disabled":
        try:
            wandb.login(key=wandb_key)
            print("W&B token loaded and login OK.")
        except Exception as exc:
            print(f"W&B login warning: {exc}")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=wandb_config,
        reinit=True,
    )
    if not args.no_wandb_watch and args.wandb_mode != "disabled":
        wandb.watch(model, log="gradients", log_freq=100)
    print(f"W&B project: {args.wandb_project}")
    print(f"W&B run: {getattr(run, 'url', 'disabled/offline')}")
    return wandb, run


def build_eval_meta_map(loader):
    dataset = getattr(loader, "dataset", None)
    sample_list = getattr(dataset, "sample_list", None) if dataset is not None else None
    meta_map = {}
    if sample_list is None:
        return meta_map

    for _, row in sample_list.iterrows():
        vid = str(row.get("video_id", ""))
        qtype = str(row.get("type", "unknown"))
        qns_key = f"{vid}_{qtype}"
        meta_map[qns_key] = {
            "video_id": vid,
            "question_type": qtype,
            "question": str(row.get("question", "")),
            "answers": [str(row.get(f"a{i}", "")) for i in range(5)],
        }
    return meta_map


def calc_detailed_metrics(model, loader, device, csv_output_path, best_acc):
    print("=== Detailed evaluation ===")
    model.eval()
    type_results = {}
    prediction_rows = []
    meta_map = build_eval_meta_map(loader)

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            ff, of, qns, ans_word, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
            ff, of = ff.to(device), of.to(device)
            q_family_id = q_family_id.to(device) if q_family_id is not None else None

            out = model(ff, of, qns, ans_word, return_aux=True, q_family_id=q_family_id)
            logits = out.get("fused_score", out["logits"])
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            targets = ans_id.numpy()

            for key, pred, target, prob_vec in zip(qns_keys, preds, targets, probs):
                meta = meta_map.get(str(key), {})
                qtype = str(meta.get("question_type", "unknown"))
                video_id = str(meta.get("video_id", str(key)))
                answers = meta.get("answers", ["", "", "", "", ""])
                if len(answers) < 5:
                    answers += [""] * (5 - len(answers))
                answers = answers[:5]

                correct_idx = int(target)
                predicted_idx = int(pred)
                is_correct = int(correct_idx == predicted_idx)
                type_results.setdefault(qtype, []).append(
                    {
                        "video_id": video_id,
                        "pred": predicted_idx,
                        "target": correct_idx,
                        "correct": bool(is_correct),
                    }
                )

                prediction_rows.append(
                    {
                        "video_id": video_id,
                        "question_type": qtype,
                        "question": str(meta.get("question", "")),
                        "correct_idx": correct_idx,
                        "predicted_idx": predicted_idx,
                        "is_correct": is_correct,
                        "confidence": float(prob_vec[predicted_idx]),
                        "a0": answers[0],
                        "prob_a0": float(prob_vec[0]),
                        "a1": answers[1],
                        "prob_a1": float(prob_vec[1]),
                        "a2": answers[2],
                        "prob_a2": float(prob_vec[2]),
                        "a3": answers[3],
                        "prob_a3": float(prob_vec[3]),
                        "a4": answers[4],
                        "prob_a4": float(prob_vec[4]),
                        "predicted_answer": answers[predicted_idx],
                        "correct_answer": answers[correct_idx],
                    }
                )

    pd.DataFrame(prediction_rows).to_csv(csv_output_path, index=False)
    print(f"Saved detailed predictions CSV: {csv_output_path}")

    metrics = {}
    mapping = [
        ("Description", "descriptive"),
        ("Explanation", "explanatory"),
        ("Predictive-Answer", "predictive"),
        ("Predictive-Reason", "predictive_reason"),
        ("Counterfactual-Answer", "counterfactual"),
        ("Counterfactual-Reason", "counterfactual_reason"),
    ]
    for name, qtype in mapping:
        rows = type_results.get(qtype, [])
        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        metrics[name] = (correct / total * 100) if total > 0 else 0.0

    def calc_hard_metric(type_ans, type_reason):
        if type_ans not in type_results or type_reason not in type_results:
            return 0.0
        ans_by_vid = {row["video_id"]: row["correct"] for row in type_results[type_ans]}
        reason_by_vid = {row["video_id"]: row["correct"] for row in type_results[type_reason]}
        common_vids = set(ans_by_vid.keys()) & set(reason_by_vid.keys())
        if not common_vids:
            return 0.0
        both_correct = sum(1 for vid in common_vids if ans_by_vid[vid] and reason_by_vid[vid])
        return both_correct / len(common_vids) * 100

    metrics["PAR"] = calc_hard_metric("predictive", "predictive_reason")
    metrics["CAR"] = calc_hard_metric("counterfactual", "counterfactual_reason")
    metrics["Acc_ALL"] = (
        metrics["Description"] + metrics["Explanation"] + metrics["PAR"] + metrics["CAR"]
    ) / 4.0
    metrics["WeightedScore_WeakPriority"] = (
        0.35 * metrics["Predictive-Reason"]
        + 0.35 * metrics["Counterfactual-Reason"]
        + 0.20 * metrics["Acc_ALL"]
        + 0.10 * float(best_acc)
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    cli_args = apply_profile(build_arg_parser().parse_args())
    setup_run_logging(cli_args.run_name)
    load_env_file(cli_args.env_file)

    if cli_args.gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    require_path("DINOv3 features", cli_args.dinov3_feature_path, ".pt")
    require_path("GDINO/FRCNN features", cli_args.gdino_feature_path, ".pkl")
    require_path("Annotations QA", cli_args.annotation_path)
    require_path("Splits", cli_args.split_dir, ".pkl")
    ensure_torch_gpu_if_needed(cli_args)
    check_environment(require_gpu=not cli_args.allow_cpu)
    import_runtime_modules()
    import_project_modules()

    set_seed(cli_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not cli_args.allow_cpu:
        raise RuntimeError("CUDA is required by default. Use --allow_cpu only for debugging.")

    model_dir = Path(cli_args.model_dir or (Path.cwd() / "models" / cli_args.run_name))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_args = build_model_config(cli_args, device)
    print("=== Config ===")
    print(f"Run name: {cli_args.run_name}")
    print(f"Run profile: {cli_args.run_profile}")
    print(f"Project: {cli_args.wandb_project}")
    print(f"Model dir: {model_dir}")
    print(f"Device: {device}")
    print(gpu_status("GPU selected"))
    print(
        "Effective bs: "
        f"physical={model_args.bs} x accum={model_args.accumulation_steps} "
        f"= {model_args.bs * model_args.accumulation_steps}"
    )

    model_filename = f"best_model_gdinofrcnn_ncod_hum_{cli_args.run_name}.ckpt"
    latest_ckpt_filename = f"latest_checkpoint_gdinofrcnn_ncod_hum_{cli_args.run_name}.ckpt"
    train_history_filename = f"train_history_gdinofrcnn_ncod_hum_{cli_args.run_name}.csv"
    predictions_csv_filename = f"test_predictions_gdinofrcnn_ncod_hum_{cli_args.run_name}.csv"
    metrics_json_filename = f"final_metrics_gdinofrcnn_ncod_hum_{cli_args.run_name}.json"

    save_path = model_dir / model_filename
    latest_ckpt_path = model_dir / latest_ckpt_filename
    train_history_csv_path = model_dir / train_history_filename
    predictions_csv_path = model_dir / predictions_csv_filename
    metrics_json_path = model_dir / metrics_json_filename
    comparison_csv_path = model_dir / "run_comparison_gdino_local.csv"

    (
        train_ds,
        _val_ds,
        _test_ds,
        train_loader,
        val_loader,
        test_loader,
        train_sample_keys,
        train_key_to_idx,
    ) = create_dataloaders(model_args)

    print("=== Model ===")
    cfg = vars(model_args).copy()
    model = VideoQAmodel(**cfg)
    model.to(device)

    non_text_params = []
    text_base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder" in name:
            text_base_params.append(param)
        else:
            non_text_params.append(param)

    param_groups = []
    if non_text_params:
        param_groups.append({"params": non_text_params, "lr": model_args.lr, "weight_decay": model_args.decay})
    if text_base_params:
        param_groups.append(
            {"params": text_base_params, "lr": model_args.text_encoder_lr, "weight_decay": model_args.decay}
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer_model.")

    optimizer_model = torch.optim.AdamW(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer_model,
        mode="max",
        factor=model_args.gamma,
        patience=model_args.lr_patience,
        threshold=model_args.early_stop_min_delta,
        threshold_mode="abs",
        min_lr=model_args.min_lr,
    )
    U = torch.nn.Parameter(torch.full((len(train_ds),), 1e-8, dtype=torch.float32, device=device))
    optimizer_u = torch.optim.SGD([U], lr=model_args.ncod_u_lr)
    bce = nn.BCEWithLogitsLoss()

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Text-encoder trainable params: {sum(p.numel() for p in text_base_params) / 1e6:.3f}M")
    print(f"U shape: {tuple(U.shape)}")

    wandb_config = {
        "run_name": cli_args.run_name,
        "run_profile": cli_args.run_profile,
        "backbone": "dinov3+groundingdino",
        "text_encoder": model_args.text_encoder_type,
        "use_lora": model_args.use_lora,
        "full_text_finetune": not model_args.freeze_text_encoder,
        "physical_batch_size": model_args.bs,
        "accumulation_steps": model_args.accumulation_steps,
        "effective_batch_size": model_args.bs * model_args.accumulation_steps,
        "epochs": model_args.epoch,
        "lambda_verifier": model_args.lambda_verifier,
        "lambda_knowledge": model_args.lambda_knowledge,
        "ncod_u_lr": model_args.ncod_u_lr,
        "hum_alpha": model_args.hum_alpha,
        "lr_main": model_args.lr,
        "lr_text_encoder": model_args.text_encoder_lr,
        "early_stop_patience": model_args.early_stop_patience,
        "early_stop_min_delta": model_args.early_stop_min_delta,
        "early_stop_start_epoch": model_args.early_stop_start_epoch,
        "dinov3_feature_path": model_args.video_feature_root,
        "gdino_feature_path": model_args.grounding_dino_path,
        "annotation_path": model_args.sample_list_path,
        "split_dir": model_args.split_dir_txt,
    }
    wandb, run = init_wandb(cli_args, model, wandb_config)

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    history_rows = []

    if cli_args.resume:
        resume_path = Path(cli_args.resume_path or latest_ckpt_path)
        print(f"Resume enabled from local checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            print(f"Warning: missing model keys when resume: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected model keys when resume: {len(unexpected)}")
        if "optimizer_model_state_dict" in checkpoint:
            optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])
        if "optimizer_u_state_dict" in checkpoint:
            optimizer_u.load_state_dict(checkpoint["optimizer_u_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "U" in checkpoint:
            with torch.no_grad():
                u_ckpt = checkpoint["U"].to(device).float().view(-1)
                n = min(u_ckpt.numel(), U.numel())
                U[:n].copy_(u_ckpt[:n])
        best_acc = float(checkpoint.get("best_acc", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = checkpoint.get("history", history)
        history_rows = checkpoint.get("history_rows", history_rows)
        print(f"Resumed. start_epoch={start_epoch} best_acc={best_acc:.2f} best_epoch={best_epoch}")
    else:
        print("Resume disabled. Training starts from epoch 1.")

    stop_training = False
    if not cli_args.skip_train:
        print("=== Training ===")
        for ep in range(start_epoch, model_args.epoch + 1):
            print(f"\nEpoch {ep}/{model_args.epoch}")
            print(gpu_status(f"GPU epoch {ep} start"))
            total_loss, l1, l2, verifier_loss, knowledge_loss, train_acc = train_epoch_integrated(
                model=model,
                optimizer_model=optimizer_model,
                optimizer_u=optimizer_u,
                U=U,
                loader=train_loader,
                bce=bce,
                device=device,
                epoch=ep,
                key_to_idx=train_key_to_idx,
                accumulation_steps=model_args.accumulation_steps,
                hum_alpha=model_args.hum_alpha,
                lambda_verifier=model_args.lambda_verifier,
                lambda_knowledge=model_args.lambda_knowledge,
            )

            val_acc = eval_epoch(model, val_loader, device)
            scheduler.step(val_acc)

            history["train_loss"].append(total_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            current_lrs = [pg["lr"] for pg in optimizer_model.param_groups]
            min_lr_now = float(min(current_lrs))
            max_lr_now = float(max(current_lrs))

            improved = val_acc > (best_acc + model_args.early_stop_min_delta)
            if improved:
                best_acc = val_acc
                best_epoch = ep
                epochs_without_improvement = 0
                print(f"New best val_acc={best_acc:.2f}% at epoch {best_epoch}")
            elif ep >= model_args.early_stop_start_epoch:
                epochs_without_improvement += 1
                print(
                    f"No significant improvement for {epochs_without_improvement} epoch(s) "
                    f"(patience={model_args.early_stop_patience}, min_delta={model_args.early_stop_min_delta})"
                )

            epoch_row = {
                "epoch": ep,
                "train_total_loss": float(total_loss),
                "train_l1": float(l1),
                "train_l2": float(l2),
                "train_verifier_loss": float(verifier_loss),
                "train_knowledge_loss": float(knowledge_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "u_mean": float(U.detach().mean().item()),
                "u_max": float(U.detach().max().item()),
                "lr_main_min": min_lr_now,
                "lr_main_max": max_lr_now,
                "best_acc_so_far": float(best_acc),
                "best_epoch_so_far": int(best_epoch),
                "epochs_without_improvement": int(epochs_without_improvement),
                "gpu_status": gpu_status(f"GPU epoch {ep} end"),
            }
            history_rows.append(epoch_row)
            pd.DataFrame(history_rows).to_csv(train_history_csv_path, index=False)
            wandb.log(epoch_row)

            ckpt = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_model_state_dict": optimizer_model.state_dict(),
                "optimizer_u_state_dict": optimizer_u.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "history_rows": history_rows,
                "U": U.detach().cpu(),
                "train_sample_keys": train_sample_keys,
                "run_name": cli_args.run_name,
            }
            torch.save(ckpt, latest_ckpt_path)

            if improved:
                torch.save(ckpt, save_path)

            print(
                "Epoch row: "
                f"loss={total_loss:.4f} train_acc={train_acc:.2f} "
                f"val_acc={val_acc:.2f} lr=[{min_lr_now:.2e},{max_lr_now:.2e}]"
            )
            print(epoch_row["gpu_status"])

            if (
                ep >= model_args.early_stop_start_epoch
                and epochs_without_improvement >= model_args.early_stop_patience
            ):
                print(f"Early stopping at epoch {ep}. Best val_acc={best_acc:.2f}% at epoch {best_epoch}.")
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_epoch"] = int(ep)
                stop_training = True
                break

        wandb.run.summary["best_val_acc"] = float(best_acc)
        wandb.run.summary["best_epoch"] = int(best_epoch)

        if save_path.exists():
            best_ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
            print(f"Loaded best checkpoint from epoch {best_epoch} for final evaluation.")

        if not stop_training:
            print(f"Training finished all {model_args.epoch} epochs. Best Val Accuracy: {best_acc:.2f}%")
    else:
        print("Skipping training.")
        resume_path = Path(cli_args.resume_path or save_path)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            best_acc = float(checkpoint.get("best_acc", 0.0))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            print(f"Loaded checkpoint for eval: {resume_path}")

    metrics = {}
    if not cli_args.skip_eval:
        metrics = calc_detailed_metrics(model, test_loader, device, predictions_csv_path, best_acc)
        metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON: {metrics_json_path}")

        comparison_row = {
            "run_name": cli_args.run_name,
            "run_profile": cli_args.run_profile,
            "best_val_acc": float(best_acc),
            "best_epoch": int(best_epoch),
            **{k: float(v) for k, v in metrics.items()},
        }
        if comparison_csv_path.exists():
            comp_df = pd.read_csv(comparison_csv_path)
            comp_df = comp_df[comp_df["run_name"] != cli_args.run_name]
            comp_df = pd.concat([comp_df, pd.DataFrame([comparison_row])], ignore_index=True)
        else:
            comp_df = pd.DataFrame([comparison_row])
        comp_df = comp_df.sort_values(by="WeightedScore_WeakPriority", ascending=False)
        comp_df.to_csv(comparison_csv_path, index=False)
        print(f"Saved/updated run comparison CSV: {comparison_csv_path}")

        wandb.log({f"eval/{k.replace('-', '_')}": float(v) for k, v in metrics.items()})
        wandb.run.summary["weighted_score_weak_priority"] = float(metrics["WeightedScore_WeakPriority"])

    if metrics_json_path.exists() or train_history_csv_path.exists():
        artifact = wandb.Artifact(
            name=f"final-results-gdinofrcnn-ncod-hum-{cli_args.run_name}",
            type="results",
            metadata={"run_name": cli_args.run_name, "run_profile": cli_args.run_profile},
        )
        for path in [metrics_json_path, predictions_csv_path, train_history_csv_path, latest_ckpt_path, save_path]:
            if path.exists():
                artifact.add_file(str(path), name=path.name)
        wandb.log_artifact(artifact)

    wandb.finish()
    print("Done.")
    print(f"Best val_acc: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Model dir: {model_dir}")
    print(f"Log dir: {Path.cwd() / 'log'}")


if __name__ == "__main__":
    main()
