# Generated from: train full_soft_routing.ipynb

# %
# CELL 1: Git Clone & Setup (Kaggle)
import os
from pathlib import Path

REPO_URL  = 'https://github.com/DanielQH07/tranSTR_Casual.git'
REPO_NAME = 'tranSTR_Casual'
BRANCH    = 'feat/generic-train-improvements'

WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO_DIR    = WORKING_DIR / REPO_NAME

def has_repo_files(p):
    p = Path(p)
    return (p / 'DataLoader.py').exists() and (p / 'networks' / 'model.py').exists()

if has_repo_files(Path.cwd()):
    print(f'Using current repo: {Path.cwd()}')
else:
    if not REPO_DIR.exists():
        print(f'Cloning {REPO_URL} (branch={BRANCH})...')
        os.system(f'git clone {REPO_URL} -b {BRANCH} {REPO_DIR}')
    target_dir = REPO_DIR / 'causalvid'
    if target_dir.exists():
        os.chdir(target_dir)
    else:
        os.chdir(REPO_DIR)
    print(f'Changed directory to: {Path.cwd()}')

if not has_repo_files(Path.cwd()):
    raise FileNotFoundError(f'Repo files not found in {Path.cwd()}. Expected DataLoader.py and networks/model.py.')

print(f'Working directory: {Path.cwd()}')

# %
# CELL 2: Dependencies + W&B login (Kaggle Secrets)
print('=== CELL 2: Dependencies & W&B Setup ===')
import os, sys, importlib.util

# Fix common Kaggle base-image conflict: pin huggingface_hub<1.0 to be compatible with preinstalled transformers.
# peft is needed only when Config.use_lora=True.
os.system('pip install -q "huggingface_hub<1.0" "transformers>=4.41,<5.0" wandb peft')

# Force-reload in case transformers/huggingface_hub were already imported with bad versions
for mod in list(sys.modules):
    if mod.startswith('transformers') or mod.startswith('huggingface_hub'):
        del sys.modules[mod]

import huggingface_hub, transformers, wandb
print(f'huggingface_hub=={huggingface_hub.__version__} | transformers=={transformers.__version__}')

# ============================================
# W&B CONFIG (use Kaggle Secrets: WANDB_API_KEY)
# ============================================
WANDB_PROJECT = 'transtr-causalvid-dino'
WANDB_ENTITY  = None

wandb_key = os.environ.get('WANDB_API_KEY')
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    wandb_key = wandb_key or secrets.get_secret('WANDB_API_KEY')
    print('Kaggle secrets checked')
except Exception as e:
    print(f'Kaggle secrets not available: {e}')

if wandb_key:
    wandb.login(key=wandb_key)
    print('W&B login OK')
else:
    os.environ.setdefault('WANDB_MODE', 'offline')
    print('WANDB_API_KEY not set; W&B will run offline')

# %
# CELL 3: Resolve data paths from /kaggle/input/ (no HF token, no KaggleHub download)
#  - DINOv3 frame features
#  - GroundingDINO + FasterRCNN object features
#  - Annotations (QA)
#  - Splits (train/val/test pkl)
print('=== CELL 3: Data Paths (Kaggle Inputs) ===')

import os, glob, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

KAGGLE_INPUT = Path('/kaggle/input')

# Default Kaggle dataset slugs/dirs. Override below if the names attached to your notebook differ.
DATASET_DEFAULTS = {
    'dinov3':   KAGGLE_INPUT / 'dinov3-feat',                  # DINOv3 frame features (.pt)
    'gdino':    KAGGLE_INPUT / 'gdinofrcnn-features',          # GDINO + FRCNN object features (.pkl)
    'anno':     KAGGLE_INPUT / 'text-annotation',              # QA annotations
    'split':    KAGGLE_INPUT / 'casual-vid-data-split',        # train/valid/test pkl
}

# Optional manual overrides (leave None to auto-detect by slug substring).
DINOV3_ROOT_OVERRIDE = None
GDINO_ROOT_OVERRIDE  = None
ANNO_ROOT_OVERRIDE   = None
SPLIT_ROOT_OVERRIDE  = None


def _resolve_root(kind, override=None, slug_hints=()):
    if override and Path(override).exists():
        return Path(override)
    default = DATASET_DEFAULTS[kind]
    if default.exists():
        return default
    if KAGGLE_INPUT.exists():
        candidates = list(KAGGLE_INPUT.iterdir())
        for hint in slug_hints:
            for p in candidates:
                if p.is_dir() and hint.lower() in p.name.lower():
                    return p
    raise FileNotFoundError(
        f'Missing Kaggle input dataset for {kind} (expected {default}). '
        f'Attach the dataset in Kaggle or set the *_OVERRIDE variable above.'
    )


def _find_dir_containing(root, target_name):
    root = Path(root)
    if not root.exists():
        return None
    if root.name.lower() == target_name.lower():
        return str(root)
    for p in root.rglob('*'):
        if p.is_dir() and p.name.lower() == target_name.lower():
            return str(p)
    return None


def _find_dir_with_ext(root, ext):
    root = Path(root)
    if not root.exists():
        return None
    counts = {}
    for p in root.rglob(f'*{ext}'):
        parent = str(p.parent)
        counts[parent] = counts.get(parent, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]


print('\n--- Resolving Kaggle input datasets ---')
dinov3_root = _resolve_root('dinov3', DINOV3_ROOT_OVERRIDE, slug_hints=['dinov3', 'dino-v3'])
gdino_root  = _resolve_root('gdino',  GDINO_ROOT_OVERRIDE,  slug_hints=['gdinofrcnn', 'gdino-frcnn', 'gdino', 'object-detection-causal'])
anno_root   = _resolve_root('anno',   ANNO_ROOT_OVERRIDE,   slug_hints=['text-annotation', 'annotation'])
split_root  = _resolve_root('split',  SPLIT_ROOT_OVERRIDE,  slug_hints=['casual-vid-data-split', 'data-split', 'split'])

print(f'DINOv3 root : {dinov3_root}')
print(f'GDINO  root : {gdino_root}')
print(f'Anno   root : {anno_root}')
print(f'Split  root : {split_root}')

# ============================================
# DINOv3 features: if dataset has split subdirs (train/val/test), merge into a single flat dir
# under /kaggle/working. Otherwise, use the dataset path directly.
# ============================================
def _has_split_subdirs(root):
    if not Path(root).exists():
        return False
    names = {p.name.lower() for p in Path(root).iterdir() if p.is_dir()}
    return ('train' in names) and ('test' in names) and ('valid' in names or 'val' in names)

candidate_roots = [Path(dinov3_root), Path(dinov3_root) / 'features']
split_source = next((c for c in candidate_roots if _has_split_subdirs(c)), None)

CLIP_MERGED_PATH = '/kaggle/working/dinov3_T16_dim1024_merge'
if split_source is not None:
    os.makedirs(CLIP_MERGED_PATH, exist_ok=True)
    print(f'\nDINOv3 split source: {split_source} → merging .pt files into {CLIP_MERGED_PATH}')
    for split in ['train', 'test', 'valid', 'val']:
        split_folder = Path(split_source) / split
        if not split_folder.exists():
            continue
        split_pt = [f for f in os.listdir(split_folder) if f.endswith('.pt')]
        print(f'  {split}: {len(split_pt)} files')
        for fname in tqdm(split_pt, desc=f'  {split}'):
            src = split_folder / fname
            dst = Path(CLIP_MERGED_PATH) / fname
            if not dst.exists():
                # Use symlink to avoid duplicating ~30GB+ data on Kaggle disk; fall back to copy.
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
    final_count = len([f for f in os.listdir(CLIP_MERGED_PATH) if f.endswith('.pt')])
    print(f'  Merge complete: {final_count} .pt files')
else:
    flat = _find_dir_with_ext(dinov3_root, '.pt')
    CLIP_MERGED_PATH = flat or str(dinov3_root)
    print(f'\nDINOv3 already flat: {CLIP_MERGED_PATH}')

# ============================================
# GDINO+FRCNN features: locate dir that contains the most .pkl files
# ============================================
GDINO_MERGED_PATH = _find_dir_with_ext(gdino_root, '.pkl') or str(gdino_root)

ANNOTATION_QA_PATH = _find_dir_containing(anno_root, 'QA') or str(anno_root)
SPLIT_TXT_PATH     = _find_dir_containing(split_root, 'split') or str(split_root)

print('\nResolved training paths:')
print(f'  CLIP_MERGED_PATH   : {CLIP_MERGED_PATH}')
print(f'  GDINO_MERGED_PATH  : {GDINO_MERGED_PATH}')
print(f'  ANNOTATION_QA_PATH : {ANNOTATION_QA_PATH}')
print(f'  SPLIT_TXT_PATH     : {SPLIT_TXT_PATH}')
print('Data paths ready.')

# %
# CELL 4: Core imports + soft routing + NCOD/HUM training functions
print('=== CELL 4: Imports + Functions (Soft Router + NCOD + HUM) ===')

import json
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from utils.util import set_seed, set_gpu_devices
from DataLoader import VideoQADataset
from networks.model import VideoQAmodel

def _unpack_batch(batch):
    if len(batch) == 7:
        ff, of, q, a, ans_id, qns_key, q_family_id = batch
    elif len(batch) == 6:
        ff, of, q, a, ans_id, qns_key = batch
        q_family_id = None
    else:
        raise ValueError(f'Unexpected batch format with {len(batch)} elements')
    return ff, of, q, a, ans_id, qns_key, q_family_id

def _compute_sample_indices(qns_keys, key_to_idx, device):
    idxs = [key_to_idx.get(str(k), -1) for k in qns_keys]
    if any(i < 0 for i in idxs):
        missing = [str(qns_keys[i]) for i, v in enumerate(idxs) if v < 0][:5]
        raise KeyError(f'Missing qns_key in key_to_idx mapping: {missing}')
    return torch.tensor(idxs, dtype=torch.long, device=device)

_QTYPE_SUFFIXES = [
    'counterfactual_reason',
    'predictive_reason',
    'counterfactual',
    'predictive',
    'explanatory',
    'descriptive',
]

def _split_qns_key(qns_key):
    key = str(qns_key)
    for qtype in _QTYPE_SUFFIXES:
        suffix = f'_{qtype}'
        if key.endswith(suffix):
            return key[:-len(suffix)], qtype
    return key, 'unknown'

def _compute_acc_all_metrics(type_results):
    mapping = [
        ('Description', 'descriptive'),
        ('Explanation', 'explanatory'),
        ('Predictive-Answer', 'predictive'),
        ('Predictive-Reason', 'predictive_reason'),
        ('Counterfactual-Answer', 'counterfactual'),
        ('Counterfactual-Reason', 'counterfactual_reason'),
    ]
    metrics = {}
    for name, qtype in mapping:
        rows = type_results.get(qtype, [])
        metrics[name] = (sum(1 for r in rows if r['correct']) / len(rows) * 100) if rows else 0.0

    def _hard_metric(type_ans, type_reason):
        ans_by_vid = {r['video_id']: r['correct'] for r in type_results.get(type_ans, [])}
        reason_by_vid = {r['video_id']: r['correct'] for r in type_results.get(type_reason, [])}
        common_vids = set(ans_by_vid) & set(reason_by_vid)
        if not common_vids:
            return 0.0
        both_correct = sum(1 for vid in common_vids if ans_by_vid[vid] and reason_by_vid[vid])
        return both_correct / len(common_vids) * 100

    metrics['PAR'] = _hard_metric('predictive', 'predictive_reason')
    metrics['CAR'] = _hard_metric('counterfactual', 'counterfactual_reason')
    metrics['Acc_ALL'] = (metrics['Description'] + metrics['Explanation'] + metrics['PAR'] + metrics['CAR']) / 4.0
    return metrics

def _hard_negative_similarity(cand_feat, target):
    """Maximum non-gold/gold cosine similarity r_j in [0, 1]."""
    if cand_feat is None:
        return torch.zeros_like(target, dtype=torch.float32)
    with torch.no_grad():
        cand_norm = F.normalize(cand_feat.detach(), dim=-1)
        gold = cand_norm.gather(1, target.view(-1, 1, 1).expand(-1, 1, cand_norm.size(-1))).squeeze(1)
        sims = torch.bmm(cand_norm, gold.unsqueeze(-1)).squeeze(-1)
        sims.scatter_(1, target.view(-1, 1), -1.0)
        return sims.max(dim=1).values.clamp(min=0.0, max=1.0)

def _hard_negative_weights(similarity, enabled=False, max_weight=1.5):
    if (not enabled) or max_weight <= 1.0:
        return torch.ones_like(similarity)
    return 1.0 + (float(max_weight) - 1.0) * similarity

class SoftUncertaintyRouter(nn.Module):
    """Lightweight sample-wise gate: 0 -> LUM, 1 -> HUM."""
    def __init__(self, d_model, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.h_norm = nn.LayerNorm(d_model)
        self.signal_norm = nn.LayerNorm(5)
        self.net = nn.Sequential(
            nn.Linear(d_model + 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, uncertainty, ce, entropy, margin, hard_negative_similarity):
        # Detach dynamic signals to avoid a shortcut through CE/entropy features.
        h = self.h_norm(h.detach())
        signals = torch.stack([
            uncertainty,
            torch.log1p(ce.clamp(0.0, 20.0)),
            entropy,
            margin,
            hard_negative_similarity,
        ], dim=1)
        z = torch.cat([h, self.signal_norm(signals.detach())], dim=1)
        return torch.sigmoid(self.net(z)).squeeze(-1)

def _update_ema_model(ema_model, model, decay):
    if ema_model is None:
        return
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

def train_epoch_integrated(
    model, router, optimizer_model, optimizer_u, U, loader, xe, bce, device, epoch, key_to_idx,
    accumulation_steps=4, hum_alpha=1.0, lambda_verifier=0.2, lambda_knowledge=0.1,
    lambda_prior=0.1, use_hard_neg_mining=False, hard_neg_weight_max=1.5,
    ema_model=None, ema_decay=0.999,
    scheduler=None, scheduler_step_per_batch=False
):
    model.train()
    router.train()
    total_loss, total_l1, total_l2 = 0.0, 0.0, 0.0
    total_verifier, total_knowledge, total_prior = 0.0, 0.0, 0.0
    gate_sum = gate_de_sum = gate_pc_sum = 0.0
    gate_count = gate_de_count = gate_pc_count = 0
    correct, total = 0, 0
    optimizer_model.zero_grad()
    optimizer_u.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, batch in enumerate(pbar):
        ff, of, q, a, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
        ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)

        if q_family_id is None:
            q_family_id = torch.zeros_like(tgt)
        else:
            q_family_id = q_family_id.to(device)

        sample_indices = _compute_sample_indices(qns_keys, key_to_idx, device)
        out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
        logits = out['logits']
        fused_logits = out.get('fused_score', logits)
        verifier_logits = out.get('verifier_logits', logits)
        knowledge_logits = out.get('knowledge_score', None)
        cand_feat = out.get('cand_feat', None)

        probs = torch.softmax(logits, dim=1)
        y_onehot = F.one_hot(tgt, num_classes=logits.size(-1)).float()
        u_batch = U[sample_indices].unsqueeze(1)

        ce_per_sample = -torch.sum(y_onehot * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        shifted_probs = torch.clamp(probs + (u_batch.detach() * y_onehot), min=1e-12, max=1.0)
        lum_loss = -torch.sum(y_onehot * torch.log(shifted_probs), dim=1)
        hum_loss = (1.0 + hum_alpha * u_batch.detach().squeeze(1)) * ce_per_sample

        # z_j = [h_j, U_j, CE_j, H(p_j), Delta_j, r_j]
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)
        entropy = entropy / math.log(max(2, logits.size(1)))
        top2 = probs.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        hard_neg_similarity = _hard_negative_similarity(cand_feat, tgt)
        gate = router(
            out['mem_pool'],
            u_batch.detach().squeeze(1),
            ce_per_sample.detach(),
            entropy.detach(),
            margin.detach(),
            hard_neg_similarity,
        )
        family_prior = (q_family_id >= 2).float()
        prior_loss = F.binary_cross_entropy(gate.clamp(1e-6, 1.0 - 1e-6), family_prior)
        routed_loss = (1.0 - gate) * lum_loss + gate * hum_loss
        hard_neg_w = _hard_negative_weights(
            hard_neg_similarity, enabled=use_hard_neg_mining, max_weight=hard_neg_weight_max
        )
        l1 = (routed_loss * hard_neg_w).mean()

        verifier_loss = bce(verifier_logits, y_onehot)
        if knowledge_logits is not None:
            knowledge_loss = bce(knowledge_logits, y_onehot)
        else:
            knowledge_loss = torch.tensor(0.0, device=device)

        model_loss = l1 + lambda_prior * prior_loss + lambda_verifier * verifier_loss + lambda_knowledge * knowledge_loss
        (model_loss / accumulation_steps).backward()

        shifted_det = probs.detach() + (u_batch * y_onehot)
        l2 = F.mse_loss(shifted_det, y_onehot)
        (l2 / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(router.parameters()), max_norm=1.0)
            optimizer_model.step()
            _update_ema_model(ema_model, model, ema_decay)
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()
            optimizer_model.zero_grad()
            optimizer_u.step()
            optimizer_u.zero_grad()
            with torch.no_grad():
                U.clamp_(0.0, 0.99)

        total_l1 += l1.item()
        total_l2 += l2.item()
        total_verifier += verifier_loss.item()
        total_knowledge += knowledge_loss.item()
        total_prior += prior_loss.item()
        total_loss += (model_loss + l2).item()
        with torch.no_grad():
            de_mask = family_prior < 0.5
            pc_mask = ~de_mask
            gate_sum += gate.sum().item()
            gate_count += gate.numel()
            if de_mask.any():
                gate_de_sum += gate[de_mask].sum().item()
                gate_de_count += int(de_mask.sum().item())
            if pc_mask.any():
                gate_pc_sum += gate[pc_mask].sum().item()
                gate_pc_count += int(pc_mask.sum().item())
        correct += (fused_logits.argmax(-1) == tgt).sum().item()
        total += tgt.size(0)

        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'route': total_l1 / (batch_idx + 1),
            'prior': total_prior / (batch_idx + 1),
            'gate': gate_sum / max(gate_count, 1),
            'acc': correct / max(total, 1) * 100
        })

    n = len(loader)
    return (
        total_loss / n,
        total_l1 / n,
        total_l2 / n,
        total_verifier / n,
        total_knowledge / n,
        total_prior / n,
        gate_sum / max(gate_count, 1),
        gate_de_sum / max(gate_de_count, 1),
        gate_pc_sum / max(gate_pc_count, 1),
        correct / max(total, 1) * 100
    )

def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    type_results = {}
    with torch.no_grad():
        for batch in loader:
            ff, of, q, a, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
            ff, of, tgt = ff.to(device), of.to(device), ans_id.to(device)
            q_family_id = q_family_id.to(device) if q_family_id is not None else None
            out = model(ff, of, q, a, return_aux=True, q_family_id=q_family_id)
            logits = out.get('fused_score', out['logits'])
            preds = logits.argmax(-1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

            for key, pred, target in zip(qns_keys, preds.detach().cpu().tolist(), tgt.detach().cpu().tolist()):
                video_id, qtype = _split_qns_key(key)
                type_results.setdefault(qtype, []).append({
                    'video_id': video_id,
                    'correct': bool(int(pred) == int(target)),
                })

    metrics = _compute_acc_all_metrics(type_results)
    metrics['Plain_Acc'] = correct / max(total, 1) * 100
    return metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print('Imports and functions defined for integrated training.')

# %
# CELL 5: Setup Paths & Config (Kaggle, bs=8 + accum=4 -> effective bs=32)
print('=== CELL 5: Paths & Config ===')

clip_merged_path   = globals().get('CLIP_MERGED_PATH', None)
gdino_merged_path  = globals().get('GDINO_MERGED_PATH', None)
annotation_qa_path = globals().get('ANNOTATION_QA_PATH', None)
split_txt_path     = globals().get('SPLIT_TXT_PATH', None)

CLIP_FEATURE_PATH  = clip_merged_path  or '/kaggle/working/dinov3_T16_dim1024_merge'
GDINO_FEATURE_PATH = gdino_merged_path or '/kaggle/input/gdinofrcnn-features'
ANNOTATION_PATH    = annotation_qa_path or '/kaggle/input/text-annotation/QA'
SPLIT_DIR          = split_txt_path    or '/kaggle/input/casual-vid-data-split/split'

BASE = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print('\n--- Path Verification ---')
def verify_path(name, path):
    if os.path.exists(path):
        items = os.listdir(path)[:3]
        print(f'OK {name}: {items}')
        return True
    print(f'NOT FOUND {name}: {path}')
    return False

all_ok = True
all_ok &= verify_path('DINOv3 Features (1024D)', CLIP_FEATURE_PATH)
all_ok &= verify_path('GDINO+FRCNN Features (2820D)', GDINO_FEATURE_PATH)
all_ok &= verify_path('Annotations (QA)', ANNOTATION_PATH)
all_ok &= verify_path('Splits', SPLIT_DIR)

if not all_ok:
    raise FileNotFoundError('One or more required data paths are missing. Re-run CELL 3 or attach Kaggle datasets.')

import glob as _glob
n_pt  = len(_glob.glob(os.path.join(CLIP_FEATURE_PATH, '*.pt')))
n_pkl = len(_glob.glob(os.path.join(GDINO_FEATURE_PATH, '*.pkl')))
print(f'\nDINOv3 .pt: {n_pt} | GDINO+FRCNN .pkl: {n_pkl}')

# ============================================
# 3-RUN TUNING PRESETS
# ============================================
RUN_TRAINING = True
RUN_PROFILE = 'run1'
RUN_VARIANT = 'weak_soft_router_lora_hn_ema_cos'
RUN3_REG_MODE = 'dropout'

RUN_PROFILES = {
    'baseline': {
        'epoch': 10, 'lr': 1e-5,
        'lambda_verifier': 0.3, 'lambda_knowledge': 0.2,
        'early_stop_start_epoch': 5, 'early_stop_patience': 4,
        'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
    },
    'run1': {
        'epoch': 10, 'lr': 1e-5,
        'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
        'early_stop_start_epoch': 5, 'early_stop_patience': 4,
        'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
    },
    'run2': {
        'epoch': 10, 'lr': 8e-6,
        'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
        'early_stop_start_epoch': 6, 'early_stop_patience': 5,
        'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
    },
    'run3': {
        'epoch': 10, 'lr': 8e-6,
        'lambda_verifier': 0.25, 'lambda_knowledge': 0.3,
        'early_stop_start_epoch': 6, 'early_stop_patience': 5,
        'dropout': 0.3, 'encoder_dropout': 0.3, 'decay': 1e-4,
    },
}

if RUN_PROFILE not in RUN_PROFILES:
    raise ValueError(f'Invalid RUN_PROFILE={RUN_PROFILE}')

RUN_TAG = f'{RUN_PROFILE}_{RUN_VARIANT}'
MODEL_FILENAME = f'best_model_gdinofrcnn_soft_router_{RUN_TAG}.ckpt'
LATEST_CKPT_FILENAME = f'latest_checkpoint_gdinofrcnn_soft_router_{RUN_TAG}.ckpt'
TRAIN_HISTORY_FILENAME = f'train_history_gdinofrcnn_soft_router_{RUN_TAG}.csv'
PREDICTIONS_CSV_FILENAME = f'test_predictions_gdinofrcnn_soft_router_{RUN_TAG}.csv'
METRICS_JSON_FILENAME = f'final_metrics_gdinofrcnn_soft_router_{RUN_TAG}.json'
BEST_ARTIFACT_NAME = f'best-model-gdinofrcnn-soft-router-{RUN_TAG}'
LAST_ARTIFACT_NAME = f'last-checkpoint-gdinofrcnn-soft-router-{RUN_TAG}'
FINAL_ARTIFACT_NAME = f'final-results-gdinofrcnn-soft-router-{RUN_TAG}'

FEAT_DIM = 1024  # DINOv3 frame
OBJ_DIM  = 2820  # FRCNN(2048) + DeBERTa cls(768) + bbox(4)
OBJ_BBOX_DIM = 4
print(f'\nBackbone: DINOv3 ({FEAT_DIM}-d frame) + GroundingDINO+FRCNN ({OBJ_DIM}-d obj, bbox split={OBJ_BBOX_DIM})')
print(f'Run profile: {RUN_PROFILE}')

class Config:
    # Paths
    video_feature_root = CLIP_FEATURE_PATH
    grounding_dino_path = GDINO_FEATURE_PATH
    sample_list_path = ANNOTATION_PATH
    split_dir_txt = SPLIT_DIR

    # Model architecture
    topK_frame = 16
    objs = 12
    frames = 16
    select_frames = 5
    topK_obj = 12

    frame_feat_dim = FEAT_DIM
    obj_feat_dim = OBJ_DIM
    use_grounding_dino = True

    # GDINO object encoding fixes
    obj_use_bbox_pos_embed = True
    obj_bbox_dim = OBJ_BBOX_DIM
    obj_hard_gather_from_frame = True

    d_model = 768
    word_dim = 768
    nheads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    normalize_before = True
    activation = 'gelu'
    dropout = 0.3
    encoder_dropout = 0.3

    # Text encoder
    text_encoder_type = 'microsoft/deberta-base'
    freeze_text_encoder = False
    text_encoder_lr = 5e-6
    text_pool_mode = 1
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = ['query_proj', 'key_proj', 'value_proj']

    # Training
    bs = 8
    accumulation_steps = 4
    lr = 1e-5
    epoch = 10
    gpu = 0
    gamma = 0.5
    decay = 1e-4
    n_query = 5
    lambda_verifier = 0.3
    lambda_knowledge = 0.2
    return_family_id = True

    # LR scheduler + early stopping
    lr_schedule = 'cosine_warmup'
    warmup_epochs = 1
    lr_patience = 1
    min_lr = 1e-7
    early_stop_patience = 4
    early_stop_min_delta = 0.05
    early_stop_start_epoch = 5

    # Generic training improvements
    use_hard_neg_mining = True
    hard_neg_weight_max = 1.5
    use_ema = True
    ema_decay = 0.999

    # NCOD + learned soft LUM/HUM routing
    ncod_u_lr = 0.1
    hum_alpha = 1.0
    router_hidden_dim = 128
    router_dropout = 0.1
    router_lr = 1e-4
    lambda_prior = 0.1

    # Aux warmup
    aux_warmup_epochs = 2

    # Other
    hard_eval = False
    pos_ratio = 1.0
    neg_ratio = 1.0
    a = 1.0
    num_workers = 4

for _k, _v in RUN_PROFILES[RUN_PROFILE].items():
    setattr(Config, _k, _v)

if RUN_PROFILE == 'run3':
    if RUN3_REG_MODE == 'dropout':
        Config.dropout = 0.25
        Config.encoder_dropout = 0.25
    elif RUN3_REG_MODE == 'decay':
        Config.decay = 8e-5
    else:
        raise ValueError(f'Invalid RUN3_REG_MODE={RUN3_REG_MODE}')

args = Config()

if args.text_encoder_type != 'microsoft/deberta-base':
    raise ValueError('Train notebook uses DeBERTa v1 only.')

set_gpu_devices(args.gpu)
set_seed(999)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
print(f'Run tag: {RUN_TAG}')
print(f'Config: frame_feat_dim={args.frame_feat_dim}, obj_feat_dim={args.obj_feat_dim}, objs={args.objs}, select_frames={args.select_frames}')
print(f'use_grounding_dino={args.use_grounding_dino} -> obj_sorter SKIPPED')
print(f'obj_use_bbox_pos_embed={args.obj_use_bbox_pos_embed} | obj_hard_gather_from_frame={args.obj_hard_gather_from_frame}')
print(f'Effective bs: physical={args.bs} x accum={args.accumulation_steps} = {args.bs * args.accumulation_steps}')
print(f'lr={args.lr} | text_encoder_lr={args.text_encoder_lr} | decay={args.decay}')
print(f'use_lora={args.use_lora} | hard_neg={args.use_hard_neg_mining} (max={args.hard_neg_weight_max}) | ema={args.use_ema} (decay={args.ema_decay})')
print(f'lr_schedule={args.lr_schedule} | warmup_epochs={args.warmup_epochs}')
print(f'aux_warmup_epochs={args.aux_warmup_epochs} | verifier={args.lambda_verifier} | knowledge={args.lambda_knowledge}')
print(f'soft_router: hidden={args.router_hidden_dim} | lr={args.router_lr} | lambda_prior={args.lambda_prior}')
print(f'Early stop: start={args.early_stop_start_epoch}, patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}')
print(f'Model file: {MODEL_FILENAME}')

# %
# CELL 6: Create Datasets
print('=== CELL 6: Datasets ===')

train_ds = VideoQADataset(
    split='train', n_query=args.n_query, obj_num=args.objs,
    sample_list_path=args.sample_list_path,
    video_feature_path=args.video_feature_root,
    grounding_dino_path=args.grounding_dino_path,
    split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
    max_samples=None, verbose=True, return_family_id=args.return_family_id
)
val_ds = VideoQADataset(
    split='val', n_query=args.n_query, obj_num=args.objs,
    sample_list_path=args.sample_list_path,
    video_feature_path=args.video_feature_root,
    grounding_dino_path=args.grounding_dino_path,
    split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
    max_samples=None, verbose=True, return_family_id=args.return_family_id
)
test_ds = VideoQADataset(
    split='test', n_query=args.n_query, obj_num=args.objs,
    sample_list_path=args.sample_list_path,
    video_feature_path=args.video_feature_root,
    grounding_dino_path=args.grounding_dino_path,
    split_dir=args.split_dir_txt, topK_frame=args.topK_frame,
    max_samples=None, verbose=True, return_family_id=args.return_family_id
)

train_loader = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_ds, args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

train_sample_keys = [f"{row.video_id}_{row.type}" for row in train_ds.sample_list.itertuples(index=False)]
train_key_to_idx = {k: i for i, k in enumerate(train_sample_keys)}

print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# %
# CELL 7: Model + Optimizers + NCOD U + Generic Improvements
print('=== CELL 7: Model ===')
cfg = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
cfg['device'] = device
cfg['topK_frame'] = args.select_frames
model = VideoQAmodel(**cfg)
model.to(device)
router = SoftUncertaintyRouter(
    d_model=args.d_model,
    hidden_dim=args.router_hidden_dim,
    dropout=args.router_dropout,
).to(device)

ema_model = None
if args.use_ema:
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

non_text_params = []
text_base_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'text_encoder' in name:
        text_base_params.append(p)
    else:
        non_text_params.append(p)

param_groups = []
if len(non_text_params) > 0:
    param_groups.append({'params': non_text_params, 'lr': args.lr, 'weight_decay': args.decay})
if len(text_base_params) > 0:
    param_groups.append({'params': text_base_params, 'lr': args.text_encoder_lr, 'weight_decay': args.decay})
param_groups.append({'params': router.parameters(), 'lr': args.router_lr, 'weight_decay': args.decay})
if len(param_groups) == 0:
    raise RuntimeError('No trainable parameters found for optimizer_model.')

optimizer_model = torch.optim.AdamW(param_groups)

if args.lr_schedule == 'cosine_warmup':
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.accumulation_steps))
    total_steps = max(1, args.epoch * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_ratio = args.min_lr / max(args.lr, 1e-12)
        return max(min_ratio, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=_lr_lambda)
else:
    scheduler = ReduceLROnPlateau(
        optimizer_model,
        mode='max',
        factor=args.gamma,
        patience=args.lr_patience,
        threshold=args.early_stop_min_delta,
        threshold_mode='abs',
        min_lr=args.min_lr
    )

U = torch.nn.Parameter(torch.full((len(train_ds),), 1e-8, dtype=torch.float32, device=device))
optimizer_u = torch.optim.SGD([U], lr=args.ncod_u_lr)

xe = nn.CrossEntropyLoss()
bce = nn.BCEWithLogitsLoss()

save_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M')
print(f'Router params: {sum(p.numel() for p in router.parameters())/1e3:.1f}K')
print(f'Text-encoder trainable params: {sum(p.numel() for p in text_base_params)/1e6:.3f}M')
print(f'EMA enabled: {ema_model is not None}')
print(f'Scheduler: {args.lr_schedule}')
print(f'U shape: {tuple(U.shape)}')
print(f'Artifacts: best={BEST_ARTIFACT_NAME} | latest={LAST_ARTIFACT_NAME}')

# %
# CELL 8: Init W&B + Resume Checkpoint
print('=== CELL 8: Initialize W&B Run ===')

start_epoch = 1
best_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0
history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
history_rows = []

LATEST_CKPT_PATH = os.path.join(MODEL_DIR, LATEST_CKPT_FILENAME)
TRAIN_HISTORY_CSV_PATH = os.path.join(MODEL_DIR, TRAIN_HISTORY_FILENAME)

RESUME_FROM_CHECKPOINT = False
RESUME_SOURCE = 'wandb'
RESUME_ARTIFACT_ALIAS = 'latest'
LOCAL_RESUME_PATH = LATEST_CKPT_PATH

wandb_config = {
    'run_tag': RUN_TAG,
    'run_profile': RUN_PROFILE,
    'run_variant': RUN_VARIANT,
    'backbone': 'dinov3+groundingdino',
    'text_encoder': args.text_encoder_type,
    'use_lora': args.use_lora,
    'lora_r': args.lora_r,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
    'full_text_finetune': not args.freeze_text_encoder,
    'physical_batch_size': args.bs,
    'accumulation_steps': args.accumulation_steps,
    'effective_batch_size': args.bs * args.accumulation_steps,
    'epochs': args.epoch,
    'lambda_verifier': args.lambda_verifier,
    'lambda_knowledge': args.lambda_knowledge,
    'ncod_u_lr': args.ncod_u_lr,
    'hum_alpha': args.hum_alpha,
    'routing': 'weakly_supervised_soft',
    'router_hidden_dim': args.router_hidden_dim,
    'router_lr': args.router_lr,
    'lambda_prior': args.lambda_prior,
    'lr_main': args.lr,
    'lr_text_encoder': args.text_encoder_lr,
    'lr_schedule': args.lr_schedule,
    'warmup_epochs': args.warmup_epochs,
    'lr_scheduler_factor': args.gamma,
    'lr_scheduler_patience': args.lr_patience,
    'min_lr': args.min_lr,
    'use_hard_neg_mining': args.use_hard_neg_mining,
    'hard_neg_weight_max': args.hard_neg_weight_max,
    'use_ema': args.use_ema,
    'ema_decay': args.ema_decay,
    'validation_metric': 'Acc_ALL_like_test',
    'early_stop_patience': args.early_stop_patience,
    'early_stop_min_delta': args.early_stop_min_delta,
    'early_stop_start_epoch': args.early_stop_start_epoch,
    'resume_enabled': RESUME_FROM_CHECKPOINT,
    'resume_source': RESUME_SOURCE,
    'platform': 'kaggle'
}

run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_TAG, config=wandb_config, reinit=True)
wandb.watch(model, log='gradients', log_freq=100)
wandb.watch(router, log='gradients', log_freq=100)
print(f'W&B run: {run.url if run else "(offline)"}')

def _load_resume_checkpoint(path, map_location):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return torch.load(path, map_location=map_location)

if RESUME_FROM_CHECKPOINT:
    print(f'Resume enabled from: {RESUME_SOURCE}')
    try:
        checkpoint = None
        resume_path = None

        if RESUME_SOURCE == 'local':
            resume_path = LOCAL_RESUME_PATH
            checkpoint = _load_resume_checkpoint(resume_path, device)
        elif RESUME_SOURCE == 'wandb':
            api = wandb.Api()
            resume_entity = WANDB_ENTITY or api.default_entity
            artifact_path = f'{resume_entity}/{WANDB_PROJECT}/{LAST_ARTIFACT_NAME}:{RESUME_ARTIFACT_ALIAS}'
            print(f'Downloading artifact: {artifact_path}')
            artifact = api.artifact(artifact_path)
            artifact_dir = artifact.download()
            candidate_path = os.path.join(artifact_dir, LATEST_CKPT_FILENAME)
            if os.path.exists(candidate_path):
                resume_path = candidate_path
            else:
                ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.ckpt')]
                if not ckpt_files:
                    raise FileNotFoundError(f'No .ckpt found in artifact folder: {artifact_dir}')
                resume_path = os.path.join(artifact_dir, ckpt_files[0])
            checkpoint = _load_resume_checkpoint(resume_path, device)
        else:
            raise ValueError("RESUME_SOURCE must be 'local' or 'wandb'")

        ckpt_state = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in ckpt_state.items():
            if k in model_state and v.shape != model_state[k].shape:
                skipped_keys.append(f'{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}')
            else:
                filtered_state[k] = v

        if skipped_keys:
            print(f'Skipped {len(skipped_keys)} keys due to shape mismatch')

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if missing:
            print(f'Warning: missing model keys when resume: {len(missing)}')
        if unexpected:
            print(f'Warning: unexpected model keys when resume: {len(unexpected)}')

        router_restored = checkpoint.get('router_state_dict') is not None
        if router_restored:
            router.load_state_dict(checkpoint['router_state_dict'])
        else:
            print('Warning: checkpoint has no router state; router starts fresh.')

        if not skipped_keys and router_restored:
            if 'optimizer_model_state_dict' in checkpoint:
                optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            if 'optimizer_u_state_dict' in checkpoint:
                optimizer_u.load_state_dict(checkpoint['optimizer_u_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if ema_model is not None and checkpoint.get('ema_model_state_dict') is not None:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
            if 'U' in checkpoint:
                with torch.no_grad():
                    u_ckpt = checkpoint['U'].to(device).float().view(-1)
                    n = min(u_ckpt.numel(), U.numel())
                    U[:n].copy_(u_ckpt[:n])

            best_acc = float(checkpoint.get('best_acc', 0.0))
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
            best_epoch = int(checkpoint.get('best_epoch', 0))
            epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', 0))
            history = checkpoint.get('history', history)
            history_rows = checkpoint.get('history_rows', history_rows)

            if os.path.exists(TRAIN_HISTORY_CSV_PATH):
                try:
                    history_rows = pd.read_csv(TRAIN_HISTORY_CSV_PATH).to_dict('records')
                    print(f'Loaded history CSV with {len(history_rows)} rows')
                except Exception as csv_err:
                    print(f'Warning: failed to load history CSV: {csv_err}')
        else:
            print('Optimizer/scheduler/U NOT restored because model/router state is incompatible.')

        print(f'Resumed from: {resume_path}')
        print(f'Start epoch: {start_epoch} | Best acc: {best_acc:.2f}% | Best epoch: {best_epoch}')
    except Exception as e:
        print(f'Warning: resume failed, starting from scratch. Error: {e}')
else:
    print('Resume disabled. Training starts from epoch 1.')

# %
# CELL 9: Integrated Training Loop + Checkpoint/CSV Logging + Early Stopping
print('=== CELL 9: Training ===')

if RUN_TRAINING:
    stop_training = False
    for ep in range(start_epoch, args.epoch + 1):
        print(f'\nEpoch {ep}/{args.epoch}')

        # Aux warmup: keep verifier/knowledge losses off in early epochs
        if ep <= args.aux_warmup_epochs:
            eff_lambda_verifier = 0.0
            eff_lambda_knowledge = 0.0
        else:
            eff_lambda_verifier = args.lambda_verifier
            eff_lambda_knowledge = args.lambda_knowledge

        (total_loss, l1, l2, verifier_loss, knowledge_loss, prior_loss,
         gate_mean, gate_de_prior_mean, gate_pc_prior_mean, train_acc) = train_epoch_integrated(
            model=model,
            router=router,
            optimizer_model=optimizer_model,
            optimizer_u=optimizer_u,
            U=U,
            loader=train_loader,
            xe=xe,
            bce=bce,
            device=device,
            epoch=ep,
            key_to_idx=train_key_to_idx,
            accumulation_steps=args.accumulation_steps,
            hum_alpha=args.hum_alpha,
            lambda_verifier=eff_lambda_verifier,
            lambda_knowledge=eff_lambda_knowledge,
            lambda_prior=args.lambda_prior,
            use_hard_neg_mining=args.use_hard_neg_mining,
            hard_neg_weight_max=args.hard_neg_weight_max,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            scheduler=scheduler,
            scheduler_step_per_batch=(args.lr_schedule == 'cosine_warmup')
        )

        eval_model = ema_model if ema_model is not None else model
        val_metrics = eval_epoch(eval_model, val_loader, device)
        val_acc = float(val_metrics['Acc_ALL'])
        val_plain_acc = float(val_metrics['Plain_Acc'])
        if args.lr_schedule != 'cosine_warmup':
            scheduler.step(val_acc)

        history['train_loss'].append(total_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        current_lrs = [pg['lr'] for pg in optimizer_model.param_groups]
        min_lr_now = float(min(current_lrs))
        max_lr_now = float(max(current_lrs))

        improved = val_acc > (best_acc + args.early_stop_min_delta)
        if improved:
            best_acc = val_acc
            best_epoch = ep
            epochs_without_improvement = 0
            print(f'New best val_acc(Acc_ALL)={best_acc:.2f}% at epoch {best_epoch} | val_plain_acc={val_plain_acc:.2f}%')
        elif ep >= args.early_stop_start_epoch:
            epochs_without_improvement += 1
            print(
                f'No significant improvement for {epochs_without_improvement} epoch(s) '
                f'(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})'
            )

        epoch_row = {
            'epoch': ep,
            'train_total_loss': float(total_loss),
            'train_l1': float(l1),
            'train_l2': float(l2),
            'train_verifier_loss': float(verifier_loss),
            'train_knowledge_loss': float(knowledge_loss),
            'train_prior_loss': float(prior_loss),
            'router_gate_mean': float(gate_mean),
            'router_gate_de_prior_mean': float(gate_de_prior_mean),
            'router_gate_pc_prior_mean': float(gate_pc_prior_mean),
            'router_gate_family_gap': float(gate_pc_prior_mean - gate_de_prior_mean),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_plain_acc': float(val_plain_acc),
            'val_Description': float(val_metrics.get('Description', 0.0)),
            'val_Explanation': float(val_metrics.get('Explanation', 0.0)),
            'val_PAR': float(val_metrics.get('PAR', 0.0)),
            'val_CAR': float(val_metrics.get('CAR', 0.0)),
            'lambda_verifier_eff': float(eff_lambda_verifier),
            'lambda_knowledge_eff': float(eff_lambda_knowledge),
            'u_mean': float(U.detach().mean().item()),
            'u_max': float(U.detach().max().item()),
            'lr_main_min': min_lr_now,
            'lr_main_max': max_lr_now,
            'best_acc_so_far': float(best_acc),
            'best_epoch_so_far': int(best_epoch),
            'epochs_without_improvement': int(epochs_without_improvement)
        }
        history_rows.append(epoch_row)
        pd.DataFrame(history_rows).to_csv(TRAIN_HISTORY_CSV_PATH, index=False)

        wandb.log(epoch_row)

        ckpt = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'router_state_dict': router.state_dict(),
            'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_u_state_dict': optimizer_u.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'epochs_without_improvement': epochs_without_improvement,
            'history': history,
            'history_rows': history_rows,
            'U': U.detach().cpu(),
            'train_sample_keys': train_sample_keys
        }

        torch.save(ckpt, LATEST_CKPT_PATH)

        if wandb.run is not None:
            last_artifact = wandb.Artifact(
                name=LAST_ARTIFACT_NAME,
                type='model',
                metadata={
                    'epoch': ep,
                    'best_acc': float(best_acc),
                    'best_epoch': int(best_epoch),
                    'val_acc': float(val_acc),
                    'val_plain_acc': float(val_plain_acc),
                    'train_acc': float(train_acc),
                    'train_total_loss': float(total_loss),
                    'epochs_without_improvement': int(epochs_without_improvement)
                }
            )
            last_artifact.add_file(LATEST_CKPT_PATH, name=LATEST_CKPT_FILENAME)
            if os.path.exists(TRAIN_HISTORY_CSV_PATH):
                last_artifact.add_file(TRAIN_HISTORY_CSV_PATH, name=TRAIN_HISTORY_FILENAME)
            wandb.log_artifact(last_artifact, aliases=['latest', f'epoch-{ep}'])

        if improved:
            torch.save(ckpt, save_path)
            if wandb.run is not None:
                best_artifact = wandb.Artifact(
                    name=BEST_ARTIFACT_NAME,
                    type='model',
                    metadata={
                        'epoch': ep,
                        'best_acc': float(best_acc),
                        'val_acc': float(val_acc),
                        'val_plain_acc': float(val_plain_acc),
                        'train_acc': float(train_acc)
                    }
                )
                best_artifact.add_file(save_path, name=MODEL_FILENAME)
                if os.path.exists(TRAIN_HISTORY_CSV_PATH):
                    best_artifact.add_file(TRAIN_HISTORY_CSV_PATH, name=TRAIN_HISTORY_FILENAME)
                wandb.log_artifact(best_artifact, aliases=['best', f'epoch-{ep}'])

        if ep >= args.early_stop_start_epoch and epochs_without_improvement >= args.early_stop_patience:
            print(f'Early stopping at epoch {ep}. Best val_acc(Acc_ALL)={best_acc:.2f}% at epoch {best_epoch}.')
            if wandb.run is not None:
                wandb.run.summary['early_stopped'] = True
                wandb.run.summary['early_stop_epoch'] = int(ep)
            stop_training = True
            break

    if wandb.run is not None:
        wandb.run.summary['best_val_acc'] = float(best_acc)
        wandb.run.summary['best_epoch'] = int(best_epoch)

    if os.path.exists(save_path):
        best_ckpt = torch.load(save_path, map_location=device)
        best_state = best_ckpt.get('ema_model_state_dict') or best_ckpt['model_state_dict']
        model.load_state_dict(best_state, strict=False)
        if best_ckpt.get('router_state_dict') is not None:
            router.load_state_dict(best_ckpt['router_state_dict'])
        if ema_model is not None and best_ckpt.get('ema_model_state_dict') is not None:
            ema_model.load_state_dict(best_ckpt['ema_model_state_dict'], strict=False)
        print(f'Loaded best checkpoint from epoch {best_epoch} for final evaluation.')

    if not stop_training:
        print(f'Training finished all {args.epoch} epochs. Best Val Acc_ALL: {best_acc:.2f}%')
else:
    print('Skipping training')

# %
# CELL 10: Detailed Evaluation + Memory Post-check + CSV export
print('=== CELL 10: Detailed Evaluation + Memory Post-check ===')
import seaborn as sns
from networks.knowledge_retriever import CausalKnowledgeRetriever

CSV_OUTPUT_PATH = os.path.join(MODEL_DIR, PREDICTIONS_CSV_FILENAME)
COMPARISON_CSV_PATH = os.path.join(MODEL_DIR, 'run_comparison_gdino_3run.csv')

TOPK_KNOWLEDGE = 5
MEMORY_PASS_THRESHOLD = 0.15
MEMORY_GATE_ENABLED = True
MEMORY_MARGIN = 0.05

def _resolve_kb_path():
    candidates = [
        os.path.join(os.getcwd(), 'data', 'causal_knowledge_bank.json'),
        '/kaggle/working/tranSTR_Casual/causalvid/data/causal_knowledge_bank.json',
        '/kaggle/working/causalvid/data/causal_knowledge_bank.json'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

KB_PATH = _resolve_kb_path()
retriever = CausalKnowledgeRetriever(KB_PATH, topk=TOPK_KNOWLEDGE) if KB_PATH else None
print(f'Knowledge bank path: {KB_PATH if KB_PATH else "NOT FOUND (memory check disabled)"}')

def _qtype_to_family(qtype):
    qtype = str(qtype)
    valid = {'descriptive', 'explanatory', 'predictive', 'predictive_reason', 'counterfactual', 'counterfactual_reason'}
    return qtype if qtype in valid else 'descriptive'

def _score_candidate_with_memory(question, candidate, q_family, video_anchors=None):
    if retriever is None:
        return 0.0, []
    hits = retriever.retrieve(
        question=str(question),
        candidate=str(candidate),
        video_anchors=video_anchors or [],
        q_family=str(q_family),
        topk=TOPK_KNOWLEDGE
    )
    top_score = max([float(h.get('score', 0.0)) for h in hits], default=0.0)
    return top_score, hits

def _build_eval_meta_map(loader):
    dataset = getattr(loader, 'dataset', None)
    sample_list = getattr(dataset, 'sample_list', None) if dataset is not None else None
    meta_map = {}
    if sample_list is None:
        return meta_map
    for _, row in sample_list.iterrows():
        vid = str(row.get('video_id', ''))
        qtype = str(row.get('type', 'unknown'))
        qns_key = f'{vid}_{qtype}'
        meta_map[qns_key] = {
            'video_id': vid,
            'question_type': qtype,
            'question': str(row.get('question', '')),
            'answers': [str(row.get(f'a{i}', '')) for i in range(5)]
        }
    return meta_map

def evaluate_detailed_v2(model, loader, device, log_to_wandb=True):
    model.eval()
    type_results = {}
    prediction_rows = []
    meta_map = _build_eval_meta_map(loader)

    memory_match_flags = []
    memory_pass_flags = []
    memory_gate_correct_flags = []

    with torch.no_grad():
        for batch in tqdm(loader):
            ff, of, qns, ans_word, ans_id, qns_keys, q_family_id = _unpack_batch(batch)
            ff, of = ff.to(device), of.to(device)
            q_family_id = q_family_id.to(device) if q_family_id is not None else None

            out = model(ff, of, qns, ans_word, return_aux=True, q_family_id=q_family_id)
            logits = out.get('fused_score', out['logits'])
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            targets = ans_id.numpy()

            for i, (key, pred, target, prob_vec) in enumerate(zip(qns_keys, preds, targets, probs)):
                meta = meta_map.get(str(key), {})
                qtype = str(meta.get('question_type', 'unknown'))
                q_family = _qtype_to_family(qtype)
                video_id = str(meta.get('video_id', str(key)))
                question = str(meta.get('question', qns[i]))
                answers = meta.get('answers', ['', '', '', '', ''])
                if len(answers) < 5:
                    answers += [''] * (5 - len(answers))
                answers = answers[:5]

                correct_idx = int(target)
                predicted_idx = int(pred)
                is_correct = int(correct_idx == predicted_idx)

                cand_mem_scores = []
                for cand in answers:
                    score, _ = _score_candidate_with_memory(question, cand, q_family, video_anchors=[video_id])
                    cand_mem_scores.append(float(score))

                mem_best_idx = int(np.argmax(cand_mem_scores)) if len(cand_mem_scores) > 0 else predicted_idx
                pred_mem_score = float(cand_mem_scores[predicted_idx]) if len(cand_mem_scores) > predicted_idx else 0.0
                gt_mem_score = float(cand_mem_scores[correct_idx]) if len(cand_mem_scores) > correct_idx else 0.0

                memory_match_pred = int(predicted_idx == mem_best_idx)
                memory_pass_pred = int(pred_mem_score >= MEMORY_PASS_THRESHOLD)

                gated_idx = predicted_idx
                if MEMORY_GATE_ENABLED and len(cand_mem_scores) > 0:
                    if (cand_mem_scores[mem_best_idx] - pred_mem_score) >= MEMORY_MARGIN:
                        gated_idx = mem_best_idx
                gated_correct = int(gated_idx == correct_idx)

                memory_match_flags.append(memory_match_pred)
                memory_pass_flags.append(memory_pass_pred)
                memory_gate_correct_flags.append(gated_correct)

                type_results.setdefault(qtype, []).append({
                    'video_id': video_id,
                    'pred': predicted_idx,
                    'target': correct_idx,
                    'correct': bool(is_correct)
                })

                prediction_rows.append({
                    'video_id': video_id,
                    'question_type': qtype,
                    'question': question,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'predicted_idx_after_memory_gate': int(gated_idx),
                    'is_correct': is_correct,
                    'is_correct_after_memory_gate': gated_correct,
                    'confidence': float(prob_vec[predicted_idx]),
                    'memory_best_idx': mem_best_idx,
                    'memory_match_pred': memory_match_pred,
                    'memory_pass_pred': memory_pass_pred,
                    'memory_pred_score': pred_mem_score,
                    'memory_gt_score': gt_mem_score,
                    'a0': answers[0], 'prob_a0': float(prob_vec[0]),
                    'a1': answers[1], 'prob_a1': float(prob_vec[1]),
                    'a2': answers[2], 'prob_a2': float(prob_vec[2]),
                    'a3': answers[3], 'prob_a3': float(prob_vec[3]),
                    'a4': answers[4], 'prob_a4': float(prob_vec[4]),
                    'predicted_answer': answers[predicted_idx],
                    'correct_answer': answers[correct_idx]
                })

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f'Saved detailed predictions CSV: {CSV_OUTPUT_PATH}')

    metrics = {}
    mapping = [
        ('Description', 'descriptive'),
        ('Explanation', 'explanatory'),
        ('Predictive-Answer', 'predictive'),
        ('Predictive-Reason', 'predictive_reason'),
        ('Counterfactual-Answer', 'counterfactual'),
        ('Counterfactual-Reason', 'counterfactual_reason')
    ]
    for name, qtype in mapping:
        rows = type_results.get(qtype, [])
        total = len(rows)
        correct = sum(1 for r in rows if r['correct'])
        metrics[name] = (correct / total * 100) if total > 0 else 0.0

    def _calc_hard_metric(type_ans, type_reason):
        if type_ans not in type_results or type_reason not in type_results:
            return 0.0
        ans_by_vid = {r['video_id']: r['correct'] for r in type_results[type_ans]}
        reason_by_vid = {r['video_id']: r['correct'] for r in type_results[type_reason]}
        common_vids = set(ans_by_vid.keys()) & set(reason_by_vid.keys())
        if len(common_vids) == 0:
            return 0.0
        both_correct = sum(1 for vid in common_vids if ans_by_vid[vid] and reason_by_vid[vid])
        return both_correct / len(common_vids) * 100

    metrics['PAR'] = _calc_hard_metric('predictive', 'predictive_reason')
    metrics['CAR'] = _calc_hard_metric('counterfactual', 'counterfactual_reason')
    metrics['Acc_ALL'] = (
        metrics['Description'] + metrics['Explanation'] + metrics['PAR'] + metrics['CAR']
    ) / 4.0

    metrics['Memory_Consistency'] = float(np.mean(memory_match_flags) * 100) if len(memory_match_flags) > 0 else 0.0
    metrics['Memory_Pass_Rate'] = float(np.mean(memory_pass_flags) * 100) if len(memory_pass_flags) > 0 else 0.0
    metrics['Memory_Gated_Acc'] = float(np.mean(memory_gate_correct_flags) * 100) if len(memory_gate_correct_flags) > 0 else 0.0

    metrics['WeightedScore_WeakPriority'] = (
        0.35 * metrics['Predictive-Reason'] +
        0.35 * metrics['Counterfactual-Reason'] +
        0.20 * metrics['Acc_ALL'] +
        0.10 * float(best_acc)
    )

    if log_to_wandb and wandb.run is not None:
        wandb.log({
            'eval/Description': metrics['Description'],
            'eval/Explanation': metrics['Explanation'],
            'eval/Predictive_Answer': metrics['Predictive-Answer'],
            'eval/Predictive_Reason': metrics['Predictive-Reason'],
            'eval/Counterfactual_Answer': metrics['Counterfactual-Answer'],
            'eval/Counterfactual_Reason': metrics['Counterfactual-Reason'],
            'eval/PAR': metrics['PAR'],
            'eval/CAR': metrics['CAR'],
            'eval/Acc_ALL': metrics['Acc_ALL'],
            'eval/Memory_Consistency': metrics['Memory_Consistency'],
            'eval/Memory_Pass_Rate': metrics['Memory_Pass_Rate'],
            'eval/Memory_Gated_Acc': metrics['Memory_Gated_Acc'],
            'eval/WeightedScore_WeakPriority': metrics['WeightedScore_WeakPriority']
        })

    print(f"PAR: {metrics['PAR']:.2f}% | CAR: {metrics['CAR']:.2f}% | Acc_ALL: {metrics['Acc_ALL']:.2f}%")
    print(f"Memory_Consistency: {metrics['Memory_Consistency']:.2f}% | Memory_Gated_Acc: {metrics['Memory_Gated_Acc']:.2f}%")
    print(f"WeightedScore_WeakPriority: {metrics['WeightedScore_WeakPriority']:.2f}")
    return metrics, type_results, CSV_OUTPUT_PATH

metrics, raw_results, predictions_csv = evaluate_detailed_v2(model, test_loader, device, log_to_wandb=True)

comparison_row = {
    'run_tag': RUN_TAG,
    'run_profile': RUN_PROFILE,
    'routing': 'weakly_supervised_soft',
    'lambda_prior': float(args.lambda_prior),
    'best_val_acc': float(best_acc),
    'best_epoch': int(best_epoch),
    'Description': float(metrics.get('Description', 0.0)),
    'Explanation': float(metrics.get('Explanation', 0.0)),
    'Predictive-Answer': float(metrics.get('Predictive-Answer', 0.0)),
    'Predictive-Reason': float(metrics.get('Predictive-Reason', 0.0)),
    'Counterfactual-Answer': float(metrics.get('Counterfactual-Answer', 0.0)),
    'Counterfactual-Reason': float(metrics.get('Counterfactual-Reason', 0.0)),
    'PAR': float(metrics.get('PAR', 0.0)),
    'CAR': float(metrics.get('CAR', 0.0)),
    'Acc_ALL': float(metrics.get('Acc_ALL', 0.0)),
    'WeightedScore_WeakPriority': float(metrics.get('WeightedScore_WeakPriority', 0.0)),
}

if os.path.exists(COMPARISON_CSV_PATH):
    comp_df = pd.read_csv(COMPARISON_CSV_PATH)
    comp_df = comp_df[comp_df['run_tag'] != RUN_TAG]
    comp_df = pd.concat([comp_df, pd.DataFrame([comparison_row])], ignore_index=True)
else:
    comp_df = pd.DataFrame([comparison_row])

comp_df = comp_df.sort_values(by='WeightedScore_WeakPriority', ascending=False)
comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
print(f'Saved/updated run comparison CSV: {COMPARISON_CSV_PATH}')
print(comp_df[['run_tag', 'run_profile', 'best_val_acc', 'Predictive-Reason', 'Counterfactual-Reason', 'Acc_ALL', 'WeightedScore_WeakPriority']])

if wandb.run is not None:
    wandb.run.summary['run_tag'] = RUN_TAG
    wandb.run.summary['run_profile'] = RUN_PROFILE
    wandb.run.summary['weighted_score_weak_priority'] = float(metrics['WeightedScore_WeakPriority'])
print(metrics)

# %
# CELL 11: Finish W&B
print('=== CELL 11: Finish W&B ===')

METRICS_JSON_PATH = os.path.join(MODEL_DIR, METRICS_JSON_FILENAME)
with open(METRICS_JSON_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Saved metrics JSON: {METRICS_JSON_PATH}')

UPLOAD_CKPT_ARTIFACTS_AT_FINISH = True
if UPLOAD_CKPT_ARTIFACTS_AT_FINISH and wandb.run is not None:
    if os.path.exists(LATEST_CKPT_PATH):
        latest_ckpt_artifact = wandb.Artifact(
            name=LAST_ARTIFACT_NAME,
            type='model',
            metadata={
                'stage': 'finish',
                'checkpoint_kind': 'latest',
                'run_tag': RUN_TAG,
                'run_profile': RUN_PROFILE,
                'text_encoder': args.text_encoder_type,
                'lora': args.use_lora,
                'ncod_hum': True,
                'soft_routing': True,
                'lambda_prior': float(args.lambda_prior)
            }
        )
        latest_ckpt_artifact.add_file(LATEST_CKPT_PATH, name=LATEST_CKPT_FILENAME)
        if os.path.exists(TRAIN_HISTORY_CSV_PATH):
            latest_ckpt_artifact.add_file(TRAIN_HISTORY_CSV_PATH, name=TRAIN_HISTORY_FILENAME)
        wandb.log_artifact(latest_ckpt_artifact, aliases=['latest', 'finish'])
        print('Uploaded latest checkpoint artifact to W&B.')
    else:
        print(f'Warning: latest checkpoint not found at {LATEST_CKPT_PATH}')

    if os.path.exists(save_path):
        best_ckpt_artifact = wandb.Artifact(
            name=BEST_ARTIFACT_NAME,
            type='model',
            metadata={
                'stage': 'finish',
                'checkpoint_kind': 'best',
                'run_tag': RUN_TAG,
                'run_profile': RUN_PROFILE,
                'text_encoder': args.text_encoder_type,
                'lora': args.use_lora,
                'ncod_hum': True,
                'soft_routing': True,
                'lambda_prior': float(args.lambda_prior)
            }
        )
        best_ckpt_artifact.add_file(save_path, name=MODEL_FILENAME)
        if os.path.exists(TRAIN_HISTORY_CSV_PATH):
            best_ckpt_artifact.add_file(TRAIN_HISTORY_CSV_PATH, name=TRAIN_HISTORY_FILENAME)
        wandb.log_artifact(best_ckpt_artifact, aliases=['best', 'finish'])
        print('Uploaded best checkpoint artifact to W&B.')
    else:
        print(f'Warning: best checkpoint not found at {save_path}')

if wandb.run is not None:
    final_artifact = wandb.Artifact(
        name=FINAL_ARTIFACT_NAME,
        type='results',
        metadata={
            'run_tag': RUN_TAG,
            'run_profile': RUN_PROFILE,
            'backbone': 'dinov3+groundingdino',
            'text_encoder': args.text_encoder_type,
            'lora': args.use_lora,
            'ncod_hum': True,
                'soft_routing': True,
                'lambda_prior': float(args.lambda_prior),
            'platform': 'kaggle'
        }
    )
    if os.path.exists(METRICS_JSON_PATH):
        final_artifact.add_file(METRICS_JSON_PATH, name=METRICS_JSON_FILENAME)
    if os.path.exists(predictions_csv):
        final_artifact.add_file(predictions_csv, name=PREDICTIONS_CSV_FILENAME)
    if os.path.exists(TRAIN_HISTORY_CSV_PATH):
        final_artifact.add_file(TRAIN_HISTORY_CSV_PATH, name=TRAIN_HISTORY_FILENAME)
    if os.path.exists(LATEST_CKPT_PATH):
        final_artifact.add_file(LATEST_CKPT_PATH, name=LATEST_CKPT_FILENAME)
    if os.path.exists(save_path):
        final_artifact.add_file(save_path, name=MODEL_FILENAME)

    comparison_csv_path = os.path.join(MODEL_DIR, 'run_comparison_gdino_3run.csv')
    if os.path.exists(comparison_csv_path):
        final_artifact.add_file(comparison_csv_path, name='run_comparison_gdino_3run.csv')

    wandb.log_artifact(final_artifact)
    wandb.finish()
print('W&B run finished with metrics, CSVs, and checkpoints artifacts.')
