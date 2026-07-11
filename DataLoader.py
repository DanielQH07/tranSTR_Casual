import torch
import os
import re
import json
import pandas as pd
import pickle as pkl
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from utils.util import transform_bb

FAMILY_TO_ID = {
    'descriptive': 0,
    'explanatory': 1,
    'predictive': 2,
    'counterfactual': 3,
    'predictive_reason': 4,
    'counterfactual_reason': 5,
}

# GroundingDINO boxes + SigLIP2 ROI features (NB3_sigLIP2_add_feature_colab).
GDINO_ROI_DIM = 768
GDINO_CLS_DIM = 768
GDINO_BBOX_DIM = 4
GDINO_OBJ_DIM = GDINO_ROI_DIM + GDINO_CLS_DIM + GDINO_BBOX_DIM


class SparseDynamicsDataset(Dataset):
    """Unique-video DINO windows for answer-agnostic future prediction."""

    def __init__(
        self,
        video_feature_path,
        video_ids,
        frame_count=16,
        context_frames=4,
        windows_per_video=4,
        feature_dim=1024,
        seed=999,
        training=True,
    ):
        super().__init__()
        if context_frames + 1 > frame_count:
            raise ValueError("context_frames + target must fit inside frame_count")
        self.video_feature_path = str(video_feature_path)
        self.frame_count = int(frame_count)
        self.context_frames = int(context_frames)
        self.windows_per_video = int(windows_per_video)
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        self.video_ids = [
            str(video_id)
            for video_id in sorted(set(map(str, video_ids)))
            if osp.exists(osp.join(self.video_feature_path, f"{video_id}.pt"))
        ]
        if not self.video_ids:
            raise ValueError("SparseDynamicsDataset found no DINO feature files")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.video_ids) * self.windows_per_video

    def _load_feature(self, video_id):
        path = osp.join(self.video_feature_path, f"{video_id}.pt")
        try:
            feature = torch.load(path, weights_only=True)
        except TypeError:
            feature = torch.load(path)
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature)
        feature = torch.as_tensor(feature).float()
        if feature.dim() != 2:
            raise ValueError(f"Expected rank-2 DINO feature at {path}")
        feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        if feature.size(-1) > self.feature_dim:
            feature = feature[:, :self.feature_dim]
        elif feature.size(-1) < self.feature_dim:
            feature = torch.nn.functional.pad(
                feature, (0, self.feature_dim - feature.size(-1))
            )
        if feature.size(0) == 0:
            feature = torch.zeros(1, self.feature_dim)
        if feature.size(0) != self.frame_count:
            indices = torch.linspace(
                0, feature.size(0) - 1, self.frame_count
            ).round().long()
            feature = feature[indices]
        return feature

    def _sample_indices(self, video_index, window_index):
        epoch = self.epoch if self.training else 0
        sample_seed = (
            self.seed
            + epoch * 1000003
            + video_index * 1009
            + window_index * 97
        )
        rng = np.random.default_rng(sample_seed)
        indices = np.sort(
            rng.choice(
                self.frame_count,
                size=self.context_frames + 1,
                replace=False,
            )
        ).astype(np.int64)
        return indices

    def __getitem__(self, index):
        video_index = int(index) // self.windows_per_video
        window_index = int(index) % self.windows_per_video
        video_id = self.video_ids[video_index]
        feature = self._load_feature(video_id)
        indices = self._sample_indices(video_index, window_index)
        context_indices = torch.from_numpy(indices[:-1]).long()
        target_index = int(indices[-1])
        time_axis = torch.linspace(0.0, 1.0, self.frame_count)
        return (
            feature[context_indices],
            time_axis[context_indices],
            feature[target_index],
            time_axis[target_index],
            video_id,
        )


class VideoQADataset(Dataset):
    def __init__(self, split, n_query=5, obj_num=10, sample_list_path="", 
                 video_feature_path="", object_feature_path="", split_dir=None, 
                 topK_frame=16, max_samples=None, verbose=True, 
                 text_feature_path=None, grounding_dino_path=None, return_family_id=False):
        """
        DataLoader with support for pre-extracted text features and GroundingDINO ROI features.
        
        Args:
            text_feature_path: Path to cached text features (optional)
            grounding_dino_path: Path to GroundingDINO ROI features (optional)
                If provided, uses GroundingDINO features instead of Faster R-CNN
        """
        super().__init__()
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        self.topK_frame = topK_frame
        self.verbose = verbose
        self.text_feature_path = text_feature_path
        self.grounding_dino_path = grounding_dino_path
        self.use_grounding_dino = grounding_dino_path is not None
        self.return_family_id = return_family_id
        
        # Load cached text features
        self.text_features = None
        self.use_cached = False
        if text_feature_path:
            text_file = osp.join(text_feature_path, f"{split}_text_features.pkl")
            if osp.exists(text_file):
                with open(text_file, 'rb') as f:
                    self.text_features = pkl.load(f)
                self.use_cached = True
                if self.verbose:
                    print(f"[{split}] Loaded {len(self.text_features)} cached text features")

        # Detect object format
        self.obj_format = self._detect_obj_format()
        if self.verbose:
            print(f"[{split}] Object feature format: {self.obj_format}")

        # Index features ONCE for O(1) access (critical for network filesystems)
        if self.use_grounding_dino:
            self.gdino_feature_map = self._scan_gdino_features()
            self.obj_feature_map = self.gdino_feature_map  # Alias for compatibility
            if self.verbose:
                print(f"[{split}] Using GroundingDINO features")
        else:
            self.obj_feature_map = self._scan_object_features()
        
        self.vit_feature_set = self._scan_video_features()
        if self.verbose:
            print(f"[{split}] Indexed {len(self.obj_feature_map)} object features, {len(self.vit_feature_set)} ViT features")

        # Load video IDs from split PKL
        valid_vids = set()
        if split_dir:
            pkl_name = 'valid' if split == 'val' else split
            pkl_path = osp.join(split_dir, f"{pkl_name}.pkl")
            
            if osp.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pkl.load(f)
                # Handle list, set, or dict
                if isinstance(data, (list, set, tuple)):
                    valid_vids = set(str(v) for v in data)
                elif isinstance(data, dict):
                    valid_vids = set(str(k) for k in data.keys())
                else:
                    valid_vids = set()
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {pkl_path}")

        # Fallback: scan sample_list_path directories
        if not valid_vids and osp.isdir(sample_list_path):
            valid_vids = {d for d in sorted(os.listdir(sample_list_path))
                         if osp.isdir(osp.join(sample_list_path, d))}

        if max_samples and len(valid_vids) > max_samples:
            valid_vids = set(sorted(valid_vids)[:max_samples])
            if self.verbose:
                print(f"[{split}] Limited to {max_samples} videos")

        # Filter by feature availability using O(1) set lookups
        vit_available = valid_vids & self.vit_feature_set
        obj_available = valid_vids & set(self.obj_feature_map.keys())
        valid_vids = vit_available & obj_available
        
        if self.verbose:
            print(f"[{split}] ViT: {len(vit_available)}, Obj: {len(obj_available)}, Both: {len(valid_vids)}")

        # Parse annotations (unchanged loop with tqdm...)
        # ... (rest of parsing logic is fine, it was below checks) ...
        # Using tqdm to show progress for large datasets
        from tqdm.auto import tqdm
        # Keep a stable sample order across sessions so resumed NCOD U remains aligned by idx.
        sorted_vids = sorted(valid_vids)
        iterator = tqdm(sorted_vids, desc=f"[{split}] Parsing annotations") if self.verbose else sorted_vids
        
        rows = []
        for vid in iterator:
            vp = osp.join(sample_list_path, vid)
            tj, aj = osp.join(vp, "text.json"), osp.join(vp, "answer.json")
            
            # Optimization: Try to open directly instead of checking exists() twice
            try:
                with open(tj, encoding="utf-8") as f:
                    td = json.load(f)
                with open(aj, encoding="utf-8") as f:
                    ad = json.load(f)
                    
                for k in ["descriptive", "explanatory", "predictive", "counterfactual"]:
                    if k in td and k in ad:
                        q, a = td[k], ad[k]
                        if "question" in q and "answer" in q and "answer" in a:
                            r = {"video_id": vid, "question": q["question"], 
                                 "answer": a["answer"], "type": k}
                            for i, c in enumerate(q["answer"]):
                                r[f"a{i}"] = c
                            rows.append(r)
                            
                        # Cùng câu hỏi với nhánh answer (predictive/counterfactual); chỉ khác candidates (reason).
                        if k in ["predictive", "counterfactual"] and "reason" in q and "reason" in a and "question" in q:
                            r = {"video_id": vid, "question": q["question"],
                                 "answer": a["reason"], "type": f"{k}_reason"}
                            for i, c in enumerate(q["reason"]):
                                r[f"a{i}"] = c
                            rows.append(r)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            except Exception as e:
                pass

        self.sample_list = pd.DataFrame(rows)
        
        # Filter by text features if using cached
        if self.use_cached:
            self.sample_list['qns_key'] = self.sample_list.apply(
                lambda x: f"{x['video_id']}_{x['type']}", axis=1)
            before = len(self.sample_list)
            self.sample_list = self.sample_list[
                self.sample_list['qns_key'].isin(self.text_features.keys())]
            if self.verbose and before != len(self.sample_list):
                print(f"[{split}] Filtered to {len(self.sample_list)} with text features")
        
        if self.verbose:
            print(f"[{split}] Final: {len(self.sample_list)} QA pairs")

    def _scan_object_features(self):
        """
        Scans the object feature directory ONCE to build a map of {video_id: full_path}.
        Handles both flat directory and 'kaggle_subdirs' structure.
        """
        mapping = {}
        if not osp.exists(self.object_feature_path):
            return mapping
            
        if self.obj_format == 'kaggle_subdirs':
            # subdirs structure: object_feature_path/subdir/video_id.pkl
            # Only scan immediate subdirectories
            for subdir in sorted(os.listdir(self.object_feature_path)):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    for fname in sorted(os.listdir(subdir_path)):
                        if fname.endswith('.pkl'):
                            vid = fname[:-4] # remove .pkl
                            mapping[vid] = osp.join(subdir_path, fname)
        else:
            # Flat structure or per-video folder: object_feature_path/video_id/ or object_feature_path/video_id.pkl
            for item in sorted(os.listdir(self.object_feature_path)):
                if item.endswith('.pkl'):
                    vid = item[:-4]
                    mapping[vid] = osp.join(self.object_feature_path, item)
                elif osp.isdir(osp.join(self.object_feature_path, item)):
                     # If it's a directory per video, check for contents? 
                     # For existing logic 'per_frame', it expects a directory
                     mapping[item] = osp.join(self.object_feature_path, item)
        return mapping

    def _scan_gdino_features(self):
        """
        Scans the GroundingDINO feature directory to build a map of {video_id: full_path}.
        Expects flat structure: grounding_dino_path/{video_id}.pkl
        """
        mapping = {}
        if not osp.exists(self.grounding_dino_path):
            return mapping
        
        for item in os.listdir(self.grounding_dino_path):
            if item.endswith('.pkl'):
                vid = item[:-4]  # remove .pkl
                mapping[vid] = osp.join(self.grounding_dino_path, item)
        
        return mapping

    def _scan_video_features(self):
        """
        Scans the video feature directory ONCE to build a set of available video IDs.
        Expects files named {video_id}.pt
        """
        available = set()
        if not osp.exists(self.video_feature_path):
            return available
        for fname in sorted(os.listdir(self.video_feature_path)):
            if fname.endswith('.pt'):
                vid = fname[:-3]  # remove .pt
                available.add(vid)
        return available

    def _detect_obj_format(self):
        if not osp.exists(self.object_feature_path):
            return 'unknown'
        # Simple heuristic based on first few items
        for item in sorted(os.listdir(self.object_feature_path))[:5]:
            item_path = osp.join(self.object_feature_path, item)
            # Kaggle tends to have "features_node_X" folders
            if osp.isdir(item_path) and ('features_node' in item or 'part' in item):
                return 'kaggle_subdirs'
        return 'per_frame'  # Default/Fallback

    def _has_object_feature(self, vid):
        return vid in self.obj_feature_map

    def _find_object_pkl(self, vid):
        return self.obj_feature_map.get(vid)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        c = self.sample_list.iloc[idx]
        vid = str(c["video_id"])
        qns = str(c["question"])
        ans_id = int(c["answer"])
        qns_key = f"{vid}_{c['type']}"
        q_family_id = FAMILY_TO_ID.get(str(c['type']), 0)

        # Load ViT features
        ff = torch.load(osp.join(self.video_feature_path, f"{vid}.pt"), weights_only=True)
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

        # Load Object features
        if self.use_grounding_dino:
            of = self._load_gdino_object_features(vid)
        else:
            of = self._load_object_features(vid)
        of = torch.nan_to_num(of.float(), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Text features - cached or raw
        if self.use_cached and qns_key in self.text_features:
            tf = self.text_features[qns_key]
            q_encoded = torch.from_numpy(tf['q_encoded']).float()   # [q_len, 768]
            q_mask = torch.from_numpy(tf['q_mask']).bool()          # [q_len]
            qa_encoded = torch.from_numpy(tf['qa_encoded']).float() # [5, qa_len, 768]
            qa_mask = torch.from_numpy(tf['qa_mask']).bool()        # [5, qa_len]

            if self.return_family_id:
                return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key, q_family_id
            return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key
        else:
            # Raw text strings for real-time encoding
            ans_word = [f"{qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]
            if self.return_family_id:
                return ff, of, qns, ans_word, ans_id, qns_key, q_family_id
            return ff, of, qns, ans_word, ans_id, qns_key

    def _load_gdino_object_features(self, vid):
        """
        Load GroundingDINO + SigLIP2 ROI features from pickle.

        Schema (per frame):
            roi_features:          [N, 768]   (SigLIP2 normalized image embedding)
            class_text_embedding:  [N, 768]   (DeBERTa-v3 CLS over label string)
            boxes_xyxy_orig:       [N, 4]     (in original image px)

        Returns:
            torch.Tensor: [topK_frame, obj_num, 1540] = 768 + 768 + 4
            (Raw concat — LayerNorm is applied inside the model.)
        """
        pkl_path = self.gdino_feature_map.get(vid)
        if not pkl_path or not osp.exists(pkl_path):
            return torch.zeros(self.topK_frame, self.obj_num, GDINO_OBJ_DIM)

        try:
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)

            frames_data = data.get('frames', [])
            orig_h = data.get('orig_h', 1080)
            orig_w = data.get('orig_w', 1920)

            # Sample frames (align with frame features)
            nf = len(frames_data)
            if nf > self.topK_frame:
                indices = np.linspace(0, nf - 1, self.topK_frame).astype(int)
            else:
                indices = list(range(nf))

            objs = []
            for idx in indices:
                frame_dict = frames_data[idx]
                roi_feats = np.asarray(frame_dict.get('roi_features',
                                       np.zeros((0, GDINO_ROI_DIM), dtype=np.float32)), dtype=np.float32)
                cls_emb = np.asarray(frame_dict.get('class_text_embedding',
                                     np.zeros((0, GDINO_CLS_DIM), dtype=np.float32)), dtype=np.float32)
                boxes_orig = np.asarray(frame_dict.get('boxes_xyxy_orig',
                                       np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)

                if roi_feats.ndim == 1:
                    roi_feats = roi_feats.reshape(1, -1)
                if cls_emb.ndim == 1:
                    cls_emb = cls_emb.reshape(1, -1)
                if boxes_orig.ndim == 1:
                    boxes_orig = boxes_orig.reshape(1, -1)

                n_det = len(roi_feats)
                roi_feats = self._fit_last_dim(roi_feats, GDINO_ROI_DIM)

                # Align cls_emb length with roi_feats: truncate/pad rather than silent zero-out
                if len(cls_emb) != n_det:
                    if len(cls_emb) > n_det:
                        cls_emb = cls_emb[:n_det]
                    else:
                        pad_n = n_det - len(cls_emb)
                        cls_emb = np.concatenate(
                            [cls_emb, np.zeros((pad_n, cls_emb.shape[-1]), dtype=np.float32)], axis=0
                        )
                cls_emb = self._fit_last_dim(cls_emb, GDINO_CLS_DIM)

                # Same for boxes
                if len(boxes_orig) != n_det:
                    if len(boxes_orig) > n_det:
                        boxes_orig = boxes_orig[:n_det]
                    else:
                        pad_n = n_det - len(boxes_orig)
                        boxes_orig = np.concatenate(
                            [boxes_orig, np.zeros((pad_n, 4), dtype=np.float32)], axis=0
                        )
                boxes_orig = self._fit_last_dim(boxes_orig, GDINO_BBOX_DIM)

                # Normalize bbox to [0, 1]
                if n_det > 0:
                    boxes_norm = boxes_orig / np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
                    boxes_norm = np.clip(boxes_norm, 0.0, 1.0)
                else:
                    boxes_norm = np.zeros((0, 4), dtype=np.float32)

                # Sanitize NaN/Inf only — let LayerNorm in the model handle scale.
                if n_det > 0:
                    roi_feats = np.nan_to_num(roi_feats, nan=0.0, posinf=0.0, neginf=0.0)
                    cls_emb = np.nan_to_num(cls_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    boxes_norm = np.nan_to_num(boxes_norm, nan=0.0, posinf=1.0, neginf=0.0)

                # Concat: [N, 768] + [N, 768] + [N, 4] = [N, 1540]
                if n_det > 0:
                    obj_feat = np.concatenate([roi_feats, cls_emb, boxes_norm], axis=-1)
                else:
                    obj_feat = np.zeros((0, GDINO_OBJ_DIM), dtype=np.float32)

                obj_feat = torch.from_numpy(obj_feat).float()

                # Pad/truncate to obj_num
                N = obj_feat.shape[0]
                if N > self.obj_num:
                    obj_feat = obj_feat[:self.obj_num]
                elif N < self.obj_num:
                    pad = torch.zeros(self.obj_num - N, GDINO_OBJ_DIM)
                    obj_feat = torch.cat([obj_feat, pad], dim=0)

                objs.append(obj_feat)

            # Pad frames if needed
            while len(objs) < self.topK_frame:
                objs.append(torch.zeros(self.obj_num, GDINO_OBJ_DIM))

            return torch.stack(objs)  # [topK_frame, obj_num, 1540]

        except Exception:
            return torch.zeros(self.topK_frame, self.obj_num, GDINO_OBJ_DIM)

    @staticmethod
    def _fit_last_dim(array, target_dim):
        """Pad or truncate feature arrays so DataLoader batches always stack."""
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        current_dim = array.shape[-1]
        if current_dim == target_dim:
            return array
        if current_dim > target_dim:
            return array[..., :target_dim]

        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_dim - current_dim)
        return np.pad(array, pad_width, mode='constant')

    def _load_object_features(self, vid):
        objs = []
        
        if self.obj_format in ('kaggle_subdirs', 'per_frame'):
            pkl_path = self._find_object_pkl(vid)
            if pkl_path and osp.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pkl.load(f)
                    # Current export stores arrays per frame under frames; retain legacy arrays.
                    frame_records = data.get('frames') if isinstance(data, dict) else None
                    if isinstance(frame_records, list) and frame_records:
                        feat_frames, bbox_frames = [], []
                        for frame in frame_records:
                            if not isinstance(frame, dict):
                                continue
                            feat = next((frame.get(k) for k in ('features', 'roi_features', 'box_features', 'region_features') if frame.get(k) is not None), None)
                            bbox = next((frame.get(k) for k in ('bboxes', 'boxes', 'boxes_xyxy', 'boxes_xyxy_orig', 'bbox') if frame.get(k) is not None), None)
                            if feat is None:
                                continue
                            feat = np.asarray(feat, dtype=np.float32)
                            bbox = np.zeros((len(feat), 4), dtype=np.float32) if bbox is None else np.asarray(bbox, dtype=np.float32)
                            feat_frames.append(feat)
                            bbox_frames.append(bbox)
                        if not feat_frames:
                            raise ValueError("nested FasterRCNN frames contain no feature arrays")
                        feats, bboxes = feat_frames, bbox_frames
                        num_frames = len(feats)
                    else:
                        feats = np.asarray(data.get("features"))
                        bboxes = np.asarray(data.get("bboxes"))
                        num_frames = feats.shape[0]
                    indices = np.linspace(0, num_frames - 1, self.topK_frame).astype(int) if num_frames > self.topK_frame else range(num_frames)
                    
                    for i in indices:
                        feat = torch.from_numpy(np.asarray(feats[i], dtype=np.float32)).float()
                        bbox = torch.from_numpy(np.asarray(bboxes[i], dtype=np.float32)).float()
                        if feat.ndim == 1:
                            feat = feat.unsqueeze(0)
                        if bbox.ndim == 1:
                            bbox = bbox.unsqueeze(0)
                        if feat.ndim > 2:
                            feat = feat.reshape(-1, feat.shape[-1])
                        if bbox.ndim > 2:
                            bbox = bbox.reshape(-1, bbox.shape[-1])
                        
                        if feat.shape[0] > self.obj_num:
                            feat, bbox = feat[:self.obj_num], bbox[:self.obj_num]
                        elif feat.shape[0] < self.obj_num:
                            p = self.obj_num - feat.shape[0]
                            feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                            bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
                        
                        bb = torch.from_numpy(transform_bb(bbox.numpy(), 640, 480)).float()
                        objs.append(torch.cat([feat, bb], -1))
                except Exception as exc:
                    if getattr(self, "verbose", False):
                        print(f"[FasterRCNN] failed to load {pkl_path}: {exc}")
        
        while len(objs) < self.topK_frame:
            objs.append(torch.zeros(self.obj_num, 2053))

        return torch.stack(objs)


if __name__ == "__main__":
    print("DataLoader ready (with cached text support)")
