"""DataLoader cho NextQA.

Tách riêng khỏi `DataLoader.py` (CausalVidQA) để giữ pipeline causal nguyên vẹn.

Khác biệt chính so với CausalVidQA:
- Annotation đọc từ 3 CSV `train/val/test.csv` (cột:
  `video, question, answer, a0..a4, qid, type`).
- Mỗi row trong CSV là 1 sample QA (CausalVidQA mỗi video có nhiều type qua text.json).
- `qns_key = f"{video}_{qid}"` (theo eval NextQA).
- Type ∈ {CW, CH, TN, TP, TC, DC, DL, DO} → family_id 0/1/2 (D / T / C).
- Object features GroundingDINO: schema giống causal (frames[i].roi_features /
  class_text_embedding / boxes_xyxy_orig + orig_h/w).
- Video ViT features `.pt` được scan đệ quy (extractvit.ipynb ghi theo cây con).
"""

from __future__ import annotations

import os
import os.path as osp
import pickle as pkl
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 0 = descriptive, 1 = temporal, 2 = causal (HUM kích hoạt khi >= 2)
NEXTQA_TYPE_TO_FAMILY: Dict[str, int] = {
    'DC': 0, 'DL': 0, 'DO': 0,
    'TN': 1, 'TP': 1, 'TC': 1,
    'CW': 2, 'CH': 2,
}

# 1024 (ROI DINOv3-L) + 768 (cls text DeBERTa) + 4 (bbox normalized)
GDINO_DIM = 1796


def scan_video_pt(root: str) -> Dict[str, str]:
    """Recursively map {video_id: pt_path} (extractvit.ipynb ghi theo cây con)."""
    mapping: Dict[str, str] = {}
    if not osp.exists(root):
        return mapping
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith('.pt'):
                mapping[fn[:-3]] = osp.join(dirpath, fn)
    return mapping


def scan_gdino(root: str) -> Dict[str, str]:
    """Map flat {video_id: pkl_path} cho GroundingDINO output."""
    mapping: Dict[str, str] = {}
    if not osp.exists(root):
        return mapping
    for fn in os.listdir(root):
        if fn.endswith('.pkl'):
            mapping[fn[:-4]] = osp.join(root, fn)
    return mapping


def _norm_vid(v) -> str:
    s = str(v).strip()
    if s.endswith('.mp4'):
        s = s[:-4]
    return s


def _l2_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class NextQADataset(Dataset):
    """NextQA dataset trả về tuple cùng format với `VideoQADataset` (causal).

    Returns (theo `return_family_id`):
        ff, of, qns, ans_word, ans_id, qns_key[, q_family_id]

    Với:
        ff:        [topK_frame, frame_feat_dim]   (DINOv3 .pt)
        of:        [topK_frame, obj_num, 1796]    (GDINO concat)
        qns:       str (question)
        ans_word:  list[str] gồm n_query câu "{q} [SEP] {ai}"
        ans_id:    int (chỉ số đáp án đúng 0..n_query-1)
        qns_key:   "{video}_{qid}"
    """

    def __init__(
        self,
        csv_path: str,
        vid_pt_map: Dict[str, str],
        gdino_map: Dict[str, str],
        n_query: int = 5,
        obj_num: int = 20,
        topK_frame: int = 16,
        return_family_id: bool = True,
        verbose: bool = True,
        split: str = 'train',
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.topK_frame = topK_frame
        self.return_family_id = return_family_id
        self.vid_pt_map = vid_pt_map
        self.gdino_map = gdino_map

        df = pd.read_csv(csv_path)
        required = ['video', 'question', 'answer', 'a0', 'a1', 'a2', 'a3', 'a4', 'qid', 'type']
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f'{csv_path} missing columns: {miss}')
        df = df[required].copy()
        df['video'] = df['video'].map(_norm_vid)
        df['qid'] = df['qid'].astype(str)
        for c in ['question', 'a0', 'a1', 'a2', 'a3', 'a4', 'type']:
            df[c] = df[c].fillna('').astype(str)
        df['answer'] = df['answer'].astype(int)
        df['qns_key'] = df['video'] + '_' + df['qid']

        before = len(df)
        df = df[df['video'].isin(vid_pt_map.keys()) & df['video'].isin(gdino_map.keys())]
        df = df.reset_index(drop=True)
        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)
        if verbose:
            print(f'[{split}] CSV={csv_path} rows={before} -> kept={len(df)}')
        self.sample_list = df

    def __len__(self) -> int:
        return len(self.sample_list)

    # ---------- feature loaders ----------

    def _load_vit(self, vid: str) -> torch.Tensor:
        path = self.vid_pt_map[vid]
        ff = torch.load(path, weights_only=True)
        if isinstance(ff, np.ndarray):
            ff = torch.from_numpy(ff)
        ff = ff.float()
        ff = torch.nan_to_num(ff, nan=0.0, posinf=0.0, neginf=0.0)
        nf = ff.shape[0]
        if nf > self.topK_frame:
            indices = np.linspace(0, nf - 1, self.topK_frame).astype(int)
            ff = ff[indices]
        elif nf < self.topK_frame:
            ff = torch.cat([ff, torch.zeros(self.topK_frame - nf, ff.shape[1])], 0)
        return ff

    def _load_gdino(self, vid: str) -> torch.Tensor:
        pkl_path = self.gdino_map.get(vid)
        if not pkl_path or not osp.exists(pkl_path):
            return torch.zeros(self.topK_frame, self.obj_num, GDINO_DIM)
        try:
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            frames_data = data.get('frames', [])
            orig_h = data.get('orig_h', 1080)
            orig_w = data.get('orig_w', 1920)
            nf = len(frames_data)
            if nf > self.topK_frame:
                indices = np.linspace(0, nf - 1, self.topK_frame).astype(int)
            else:
                indices = list(range(nf))

            objs = []
            for idx in indices:
                fd = frames_data[idx]
                roi = np.asarray(fd.get('roi_features', np.zeros((0, 1024), np.float32)), np.float32)
                cls = np.asarray(fd.get('class_text_embedding', np.zeros((0, 768), np.float32)), np.float32)
                box = np.asarray(fd.get('boxes_xyxy_orig', np.zeros((0, 4), np.float32)), np.float32)
                n_det = len(roi)
                if len(cls) != n_det:
                    cls = np.zeros((n_det, 768), np.float32)
                if len(box) > 0:
                    box = box / np.array([orig_w, orig_h, orig_w, orig_h], np.float32)
                else:
                    box = np.zeros((n_det, 4), np.float32) if n_det > 0 else np.zeros((0, 4), np.float32)

                if n_det > 0:
                    roi = np.nan_to_num(roi)
                    cls = np.nan_to_num(cls)
                    box = np.nan_to_num(np.clip(box, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
                    roi = _l2_norm(roi)
                    cls = _l2_norm(cls)
                    feat = np.concatenate([roi, cls, box], -1)
                else:
                    feat = np.zeros((0, GDINO_DIM), np.float32)
                feat = torch.from_numpy(feat).float()

                N = feat.shape[0]
                if N > self.obj_num:
                    feat = feat[:self.obj_num]
                elif N < self.obj_num:
                    pad = torch.zeros(self.obj_num - N, GDINO_DIM)
                    feat = torch.cat([feat, pad], 0)
                objs.append(feat)

            while len(objs) < self.topK_frame:
                objs.append(torch.zeros(self.obj_num, GDINO_DIM))
            return torch.stack(objs)
        except Exception:
            return torch.zeros(self.topK_frame, self.obj_num, GDINO_DIM)

    # ---------- main ----------

    def __getitem__(self, idx: int):
        c = self.sample_list.iloc[idx]
        vid = str(c['video'])
        qns = str(c['question'])
        ans_id = int(c['answer'])
        qtype = str(c['type'])
        qns_key = str(c['qns_key'])
        q_family_id = NEXTQA_TYPE_TO_FAMILY.get(qtype, 0)

        ff = self._load_vit(vid)
        of = self._load_gdino(vid)
        of = torch.nan_to_num(of.float(), nan=0.0, posinf=0.0, neginf=0.0)

        ans_word = [f"{qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]

        if self.return_family_id:
            return ff, of, qns, ans_word, ans_id, qns_key, q_family_id
        return ff, of, qns, ans_word, ans_id, qns_key


if __name__ == '__main__':
    print('NextQADataset ready.')
