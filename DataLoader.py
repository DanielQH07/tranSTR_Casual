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
                 text_feature_path=None):
        """
        DataLoader with support for pre-extracted text features.
        
        Args:
            text_feature_path: Path to cached text features (optional)
                If provided, returns tensor features instead of raw text
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

        # Load video IDs from split
        valid_vids = set()
        if split_dir:
            pkl_name = 'valid' if split == 'val' else split
            pkl_path = osp.join(split_dir, f"{pkl_name}.pkl")
            txt_path = osp.join(split_dir, f"{pkl_name}.txt")
            
            if osp.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pkl.load(f)
                valid_vids = set(data) if isinstance(data, (list, set)) else set(data.keys())
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {pkl_path}")
            elif osp.exists(txt_path):
                with open(txt_path) as f:
                    valid_vids = {l.strip() for l in f if l.strip()}

        if not valid_vids and osp.isdir(sample_list_path):
            valid_vids = {d for d in os.listdir(sample_list_path) 
                         if osp.isdir(osp.join(sample_list_path, d))}

        if max_samples and len(valid_vids) > max_samples:
            valid_vids = set(list(valid_vids)[:max_samples])
            if self.verbose:
                print(f"[{split}] Limited to {max_samples} videos")

        # Check feature availability
        vit_available = {vid for vid in valid_vids 
                        if osp.exists(osp.join(self.video_feature_path, f"{vid}.pt"))}
        obj_available = {vid for vid in valid_vids if self._has_object_feature(vid)}
        valid_vids = vit_available & obj_available
        
        if self.verbose:
            print(f"[{split}] ViT: {len(vit_available)}, Obj: {len(obj_available)}, Both: {len(valid_vids)}")

        # Parse annotations
        # Using tqdm to show progress for large datasets
        from tqdm.auto import tqdm
        iterator = tqdm(valid_vids, desc=f"[{split}] Parsing annotations") if self.verbose else valid_vids
        
        rows = []
        for vid in iterator:
            vp = osp.join(sample_list_path, vid)
            tj, aj = osp.join(vp, "text.json"), osp.join(vp, "answer.json")
            
            # Optimization: Try to open directly instead of checking exists() twice
            # This is faster on network filesystems (Kaggle/Colab)
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
                # Unexpected errors
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

    def _detect_obj_format(self):
        if not osp.exists(self.object_feature_path):
            return 'unknown'
        for item in os.listdir(self.object_feature_path)[:5]:
            item_path = osp.join(self.object_feature_path, item)
            if osp.isdir(item_path) and 'features_node' in item:
                return 'kaggle_subdirs'
        return 'per_frame'

    def _has_object_feature(self, vid):
        if self.obj_format == 'kaggle_subdirs':
            for subdir in os.listdir(self.object_feature_path):
                if osp.exists(osp.join(self.object_feature_path, subdir, f"{vid}.pkl")):
                    return True
            return False
        vid_dir = osp.join(self.object_feature_path, vid)
        return osp.isdir(vid_dir) and any(f.endswith('.pkl') for f in os.listdir(vid_dir))

    def _find_object_pkl(self, vid):
        if self.obj_format == 'kaggle_subdirs':
            for subdir in os.listdir(self.object_feature_path):
                pkl_path = osp.join(self.object_feature_path, subdir, f"{vid}.pkl")
                if osp.exists(pkl_path):
                    return pkl_path
        return osp.join(self.object_feature_path, vid)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        c = self.sample_list.iloc[idx]
        vid = str(c["video_id"])
        qns = str(c["question"])
        ans_id = int(c["answer"])
        qns_key = f"{vid}_{c['type']}"

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
        of = self._load_object_features(vid)
        
        # Text features - cached or raw
        if self.use_cached and qns_key in self.text_features:
            tf = self.text_features[qns_key]
            q_encoded = torch.from_numpy(tf['q_encoded']).float()   # [q_len, 768]
            q_mask = torch.from_numpy(tf['q_mask']).bool()          # [q_len]
            qa_encoded = torch.from_numpy(tf['qa_encoded']).float() # [5, qa_len, 768]
            qa_mask = torch.from_numpy(tf['qa_mask']).bool()        # [5, qa_len]
            
            return ff, of, q_encoded, q_mask, qa_encoded, qa_mask, ans_id, qns_key
        else:
            # Raw text strings for real-time encoding
            ans_word = [f"{qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]
            return ff, of, qns, ans_word, ans_id, qns_key

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