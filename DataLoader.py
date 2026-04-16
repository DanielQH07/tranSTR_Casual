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


class VideoQADataset(Dataset):
    def __init__(self, split, n_query=5, obj_num=10, sample_list_path="", 
                 video_feature_path="", object_feature_path="", split_dir=None, 
                 topK_frame=16, max_samples=None, verbose=True, 
                 text_feature_path=None, grounding_dino_path=None,
                 text_annotation_path=None, qtype=-1, qtype_subset=None,
                 stage_mode="qa", chain_data_root=None):
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
        self.qtype = qtype
        self.stage_mode = stage_mode
        self.chain_data_root = chain_data_root
        self.chain_cache = {}
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        self.topK_frame = topK_frame
        self.verbose = verbose
        self.text_feature_path = text_feature_path
        self.grounding_dino_path = grounding_dino_path
        self.use_grounding_dino = grounding_dino_path is not None

        # Backward-compatible annotation/split root resolution.
        self.annotation_root = text_annotation_path if text_annotation_path else sample_list_path
        if split_dir is None and sample_list_path and osp.isdir(sample_list_path):
            split_candidate = osp.join(sample_list_path, 'valid.pkl' if split == 'val' else f'{split}.pkl')
            if osp.exists(split_candidate):
                split_dir = sample_list_path

        self.qtype_subset = self._normalize_qtype_subset(qtype_subset)
        
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
        if not valid_vids and osp.isdir(self.annotation_root):
            valid_vids = {d for d in os.listdir(self.annotation_root)
                         if osp.isdir(osp.join(self.annotation_root, d))}

        if max_samples and len(valid_vids) > max_samples:
            valid_vids = set(list(valid_vids)[:max_samples])
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
        iterator = tqdm(valid_vids, desc=f"[{split}] Parsing annotations") if self.verbose else valid_vids
        
        rows = []
        for vid in iterator:
            vp = osp.join(self.annotation_root, vid)
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
                            
                        if k in ["predictive", "counterfactual"] and "reason" in q and "reason" in a:
                            r = {"video_id": vid, "question": "Why?", 
                                 "answer": a["reason"], "type": f"{k}_reason"}
                            for i, c in enumerate(q["reason"]):
                                r[f"a{i}"] = c
                            rows.append(r)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            except Exception as e:
                pass

        self.sample_list = pd.DataFrame(rows)

        if len(self.sample_list) > 0:
            if self.qtype != -1:
                qtype_name = self._qtype_name_from_idx(self.qtype)
                if qtype_name is not None:
                    self.sample_list = self.sample_list[self.sample_list['type'] == qtype_name]

            if self.qtype_subset:
                self.sample_list = self.sample_list[self.sample_list['type'].isin(self.qtype_subset)]
        
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

    def _qtype_name_from_idx(self, idx):
        idx_to_type = {
            0: 'descriptive',
            1: 'explanatory',
            2: 'predictive',
            3: 'predictive_reason',
            4: 'counterfactual',
            5: 'counterfactual_reason',
        }
        return idx_to_type.get(int(idx))

    def _normalize_qtype_subset(self, qtype_subset):
        if qtype_subset is None:
            return None

        if isinstance(qtype_subset, str):
            items = [x.strip() for x in qtype_subset.split(',') if x.strip()]
        else:
            items = list(qtype_subset)

        normalized = []
        valid_types = {
            'descriptive', 'explanatory', 'predictive', 'predictive_reason',
            'counterfactual', 'counterfactual_reason'
        }
        for item in items:
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                mapped = self._qtype_name_from_idx(int(item))
                if mapped:
                    normalized.append(mapped)
            else:
                item_str = str(item).strip()
                if item_str in valid_types:
                    normalized.append(item_str)

        return sorted(set(normalized)) if normalized else None

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
            for subdir in os.listdir(self.object_feature_path):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    for fname in os.listdir(subdir_path):
                        if fname.endswith('.pkl'):
                            vid = fname[:-4] # remove .pkl
                            mapping[vid] = osp.join(subdir_path, fname)
        else:
            # Flat structure or per-video folder: object_feature_path/video_id/ or object_feature_path/video_id.pkl
            for item in os.listdir(self.object_feature_path):
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
        for fname in os.listdir(self.video_feature_path):
            if fname.endswith('.pt'):
                vid = fname[:-3]  # remove .pt
                available.add(vid)
        return available

    def _detect_obj_format(self):
        if not osp.exists(self.object_feature_path):
            return 'unknown'
        # Simple heuristic based on first few items
        for item in os.listdir(self.object_feature_path)[:5]:
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
        chain_text, chain_valid = self._load_chain_target(vid, str(c['type']))

        # Load ViT features
        ff = torch.load(osp.join(self.video_feature_path, f"{vid}.pt"), weights_only=True)
        if isinstance(ff, np.ndarray):
            ff = torch.from_numpy(ff)
        ff = ff.float()
        
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
        
        # Text features - cached or raw
        if self.use_cached and qns_key in self.text_features:
            tf = self.text_features[qns_key]
            q_encoded = torch.from_numpy(tf['q_encoded']).float()   # [q_len, 768]
            q_mask = torch.from_numpy(tf['q_mask']).bool()          # [q_len]
            qa_encoded = torch.from_numpy(tf['qa_encoded']).float() # [5, qa_len, 768]
            qa_mask = torch.from_numpy(tf['qa_mask']).bool()        # [5, qa_len]
            
            if self.stage_mode == "stage1_chain":
                return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key, chain_text, chain_valid
            return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key
        else:
            # Raw text strings for real-time encoding
            ans_word = [f"{qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]
            if self.stage_mode == "stage1_chain":
                return ff, of, qns, ans_word, ans_id, qns_key, chain_text, chain_valid
            return ff, of, qns, ans_word, ans_id, qns_key

    def _resolve_chain_file(self, vid):
        if not self.chain_data_root:
            return None

        candidates = [
            osp.join(self.chain_data_root, self.split, f"{vid}.json"),
            osp.join(self.chain_data_root, f"{vid}.json"),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return None

    def _load_chain_json(self, vid):
        if vid in self.chain_cache:
            return self.chain_cache[vid]

        path = self._resolve_chain_file(vid)
        if not path:
            self.chain_cache[vid] = None
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chain_cache[vid] = data
            return data
        except Exception:
            self.chain_cache[vid] = None
            return None

    def _load_chain_target(self, vid, sample_type):
        data = self._load_chain_json(vid)
        if not data:
            return "", 0

        if sample_type == 'explanatory':
            block = data.get('explanatory')
        elif sample_type in ('predictive', 'predictive_reason'):
            block = data.get('predictive')
        else:
            return "", 0

        if not isinstance(block, dict):
            return "", 0

        parts = []
        fact_observation = block.get('fact_observation')
        if isinstance(fact_observation, str) and fact_observation.strip():
            parts.append(fact_observation.strip())

        answer_chain = block.get('answer_chain')
        if isinstance(answer_chain, dict):
            final_hyp = answer_chain.get('final_hypothesis')
            if isinstance(final_hyp, str) and final_hyp.strip():
                parts.append(final_hyp.strip())

        reason_chain = block.get('reason_chain')
        if isinstance(reason_chain, dict):
            reason_hyp = reason_chain.get('final_hypothesis')
            if isinstance(reason_hyp, str) and reason_hyp.strip():
                parts.append(reason_hyp.strip())

        if not parts:
            return "", 0
        return " [SEP] ".join(parts), 1

    def _load_gdino_object_features(self, vid):
        """
        Load GroundingDINO ROI features from pickle.
        
        Returns:
            torch.Tensor: [topK_frame, obj_num, 1028] where 1028 = 1024 (ROI) + 4 (bbox normalized)
        """
        pkl_path = self.gdino_feature_map.get(vid)
        if not pkl_path or not osp.exists(pkl_path):
            return torch.zeros(self.topK_frame, self.obj_num, 1028)
        
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
                indices = range(nf)
            
            objs = []
            for idx in indices:
                frame_dict = frames_data[idx]
                roi_feats = frame_dict.get('roi_features', np.zeros((0, 1024), dtype=np.float32))  # [N, 1024]
                boxes_orig = frame_dict.get('boxes_xyxy_orig', np.zeros((0, 4), dtype=np.float32))  # [N, 4]
                
                # Normalize bbox to [0, 1]
                if len(boxes_orig) > 0:
                    boxes_norm = boxes_orig / np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
                else:
                    boxes_norm = boxes_orig
                
                # Concat: [N, 1024] + [N, 4] = [N, 1028]
                if len(roi_feats) > 0:
                    obj_feat = np.concatenate([roi_feats, boxes_norm], axis=-1)
                else:
                    obj_feat = np.zeros((0, 1028), dtype=np.float32)
                
                obj_feat = torch.from_numpy(obj_feat).float()
                
                # Pad/truncate to obj_num
                N = obj_feat.shape[0]
                if N > self.obj_num:
                    obj_feat = obj_feat[:self.obj_num]
                elif N < self.obj_num:
                    pad = torch.zeros(self.obj_num - N, 1028)
                    obj_feat = torch.cat([obj_feat, pad], dim=0)
                
                objs.append(obj_feat)
            
            # Pad frames if needed
            while len(objs) < self.topK_frame:
                objs.append(torch.zeros(self.obj_num, 1028))
            
            return torch.stack(objs)  # [topK_frame, obj_num, 1028]
        
        except Exception as e:
            # Fallback to zeros on error
            return torch.zeros(self.topK_frame, self.obj_num, 1028)

    def _load_object_features(self, vid):
        objs = []
        
        if self.obj_format == 'kaggle_subdirs':
            pkl_path = self._find_object_pkl(vid)
            if pkl_path and osp.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pkl.load(f)
                    feats = np.array(data.get('features'))
                    bboxes = np.array(data.get('bboxes'))
                    
                    num_frames = feats.shape[0]
                    indices = np.linspace(0, num_frames - 1, self.topK_frame).astype(int) if num_frames > self.topK_frame else range(num_frames)
                    
                    for i in indices:
                        feat = torch.from_numpy(feats[i]).float()
                        bbox = torch.from_numpy(bboxes[i]).float()
                        
                        if feat.shape[0] > self.obj_num:
                            feat, bbox = feat[:self.obj_num], bbox[:self.obj_num]
                        elif feat.shape[0] < self.obj_num:
                            p = self.obj_num - feat.shape[0]
                            feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                            bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
                        
                        bb = torch.from_numpy(transform_bb(bbox.numpy(), 640, 480)).float()
                        objs.append(torch.cat([feat, bb], -1))
                except:
                    pass
        
        while len(objs) < self.topK_frame:
            objs.append(torch.zeros(self.obj_num, 2053))

        return torch.stack(objs)


if __name__ == "__main__":
    print("DataLoader ready (with cached text support)")