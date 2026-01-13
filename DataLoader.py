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
                 topK_frame=16, max_samples=None, verbose=True):
        """
        Optimized DataLoader - reads directly from original PKL files (no splitting needed)
        
        Args:
            split: 'train', 'val', or 'test'
            n_query: number of answer choices (5 for multiple choice)
            obj_num: number of objects per frame
            sample_list_path: path to annotation directory
            video_feature_path: path to ViT features (.pt files)
            object_feature_path: path to object features - can be:
                - Directory with {video_id}/*.pkl (old format, per-frame)
                - Directory with subdirs containing {video_id}.pkl (new Kaggle format)
            split_dir: path to split files (train.pkl/txt, valid.pkl/txt, test.pkl/txt)
            topK_frame: number of frames to sample
            max_samples: max videos to load (None = no limit, useful for debugging)
            verbose: print detailed logs
        """
        super().__init__()
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        self.topK_frame = topK_frame
        self.verbose = verbose

        # Detect object feature format
        self.obj_format = self._detect_obj_format()
        if self.verbose:
            print(f"[{split}] Object feature format: {self.obj_format}")

        # 1. Load Split Video IDs
        valid_vids = set()
        if split_dir:
            pkl_name = 'valid' if split == 'val' else split
            pkl_path = osp.join(split_dir, f"{pkl_name}.pkl")
            txt_path = osp.join(split_dir, f"{pkl_name}.txt")
            
            if osp.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pkl.load(f)
                if isinstance(data, (list, set)):
                    valid_vids = set(data)
                elif isinstance(data, dict):
                    valid_vids = set(data.keys())
                else:
                    valid_vids = set(data)
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {pkl_path}")
            elif osp.exists(txt_path):
                with open(txt_path) as f:
                    valid_vids = {l.strip() for l in f if l.strip()}
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {txt_path}")

        # Fallback: use annotation directories
        if not valid_vids and osp.isdir(sample_list_path):
            valid_vids = {d for d in os.listdir(sample_list_path) 
                         if osp.isdir(osp.join(sample_list_path, d))}
            if self.verbose:
                print(f"[{split}] Fallback: {len(valid_vids)} videos from annotations")

        # Apply max_samples limit
        if max_samples and len(valid_vids) > max_samples:
            valid_vids = set(list(valid_vids)[:max_samples])
            if self.verbose:
                print(f"[{split}] Limited to {max_samples} videos")

        # 2. Check feature availability
        vit_available = set()
        obj_available = set()
        
        for vid in valid_vids:
            # Check ViT
            vit_path = osp.join(self.video_feature_path, self.split, f"{vid}.pt")
            if osp.exists(vit_path):
                vit_available.add(vid)
            
            # Check Object based on format
            if self._has_object_feature(vid):
                obj_available.add(vid)
        
        both_available = vit_available & obj_available
        
        if self.verbose:
            print(f"[{split}] ViT: {len(vit_available)}, Obj: {len(obj_available)}, Both: {len(both_available)}")

        valid_vids = both_available

        # 3. Parse Annotations
        rows = []
        for vid in valid_vids:
            vp = osp.join(sample_list_path, vid)
            tj, aj = osp.join(vp, "text.json"), osp.join(vp, "answer.json")
            
            if not (osp.exists(tj) and osp.exists(aj)):
                continue
                
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
            except Exception as e:
                if self.verbose:
                    print(f"[{split}] Error parsing {vid}: {e}")

        self.sample_list = pd.DataFrame(rows)
        
        if self.verbose:
            print(f"[{split}] Final: {len(self.sample_list)} QA pairs from {len(valid_vids)} videos")
            if len(self.sample_list) > 0:
                type_counts = self.sample_list['type'].value_counts()
                for qtype, count in type_counts.items():
                    print(f"    {qtype}: {count}")

    def _detect_obj_format(self):
        """Detect object feature storage format"""
        if not osp.exists(self.object_feature_path):
            return 'unknown'
        
        items = os.listdir(self.object_feature_path)
        
        # Check for subdirectories with PKL files (Kaggle format)
        for item in items[:5]:
            item_path = osp.join(self.object_feature_path, item)
            if osp.isdir(item_path):
                sub_items = os.listdir(item_path)
                # If subdir contains .pkl files directly (per-video format)
                if any(f.endswith('.pkl') for f in sub_items):
                    # Check if it's Kaggle format (features_node_X) or old format (video_id dirs)
                    if 'features_node' in item or any(f.endswith('.pkl') and '_' in f for f in sub_items):
                        return 'kaggle_subdirs'  # PKL in subdirs like features_node_X/video.pkl
                    else:
                        return 'per_frame'  # video_id/0.pkl, 1.pkl, ...
        
        return 'per_frame'  # Default

    def _has_object_feature(self, vid):
        """Check if object features exist for a video"""
        if self.obj_format == 'kaggle_subdirs':
            # Search in subdirectories
            for subdir in os.listdir(self.object_feature_path):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    pkl_path = osp.join(subdir_path, f"{vid}.pkl")
                    if osp.exists(pkl_path):
                        return True
            return False
        else:
            # Old format: object_feature_path/video_id/
            vid_dir = osp.join(self.object_feature_path, vid)
            return osp.isdir(vid_dir) and any(f.endswith('.pkl') for f in os.listdir(vid_dir))

    def _find_object_pkl(self, vid):
        """Find the pkl file for a video"""
        if self.obj_format == 'kaggle_subdirs':
            for subdir in os.listdir(self.object_feature_path):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    pkl_path = osp.join(subdir_path, f"{vid}.pkl")
                    if osp.exists(pkl_path):
                        return pkl_path
            return None
        else:
            return osp.join(self.object_feature_path, vid)  # Return directory

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        c = self.sample_list.iloc[idx]
        vid = str(c["video_id"])
        qns = str(c["question"])
        ans_id = int(c["answer"])
        ans_word = [f"[CLS] {qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]

        # 1. Load ViT features
        vit_path = osp.join(self.video_feature_path, self.split, f"{vid}.pt")
        ff = torch.load(vit_path, weights_only=True)
        if isinstance(ff, np.ndarray):
            ff = torch.from_numpy(ff)
        ff = ff.float()
        
        nf = ff.shape[0]
        if nf > self.topK_frame:
            indices = np.linspace(0, nf - 1, self.topK_frame).astype(int)
            ff = ff[indices]
        elif nf < self.topK_frame:
            pad = torch.zeros(self.topK_frame - nf, ff.shape[1])
            ff = torch.cat([ff, pad], 0)

        # 2. Load Object features
        of = self._load_object_features(vid)
        
        qns_key = f"{vid}_{c['type']}"
        return ff, of, qns, ans_word, ans_id, qns_key

    def _load_object_features(self, vid):
        """Load object features - handles both Kaggle and old format"""
        objs = []
        
        if self.obj_format == 'kaggle_subdirs':
            # NEW: Direct read from Kaggle PKL (shape: [16, 20, 2048])
            pkl_path = self._find_object_pkl(vid)
            if pkl_path and osp.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pkl.load(f)
                    
                    feats = data.get('features')  # [16, 20, 2048]
                    bboxes = data.get('bboxes')   # [16, 20, 4]
                    
                    if feats is not None and bboxes is not None:
                        if not isinstance(feats, np.ndarray):
                            feats = np.array(feats)
                        if not isinstance(bboxes, np.ndarray):
                            bboxes = np.array(bboxes)
                        
                        num_frames = feats.shape[0]
                        
                        # Sample frames to topK_frame
                        if num_frames > self.topK_frame:
                            indices = np.linspace(0, num_frames - 1, self.topK_frame).astype(int)
                        else:
                            indices = list(range(num_frames))
                        
                        for i in indices:
                            feat = torch.from_numpy(feats[i]).float()  # [20, 2048]
                            bbox = torch.from_numpy(bboxes[i]).float()  # [20, 4]
                            
                            # Limit/pad objects
                            if feat.shape[0] > self.obj_num:
                                feat = feat[:self.obj_num]
                                bbox = bbox[:self.obj_num]
                            elif feat.shape[0] < self.obj_num:
                                p = self.obj_num - feat.shape[0]
                                feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                                bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
                            
                            # Transform bbox and concat
                            bb = torch.from_numpy(transform_bb(bbox.numpy(), 640, 480)).float()
                            objs.append(torch.cat([feat, bb], -1))  # [obj_num, 2053]
                            
                except Exception as e:
                    pass  # Will pad with zeros below
        
        else:
            # OLD: Per-frame PKL format
            od = osp.join(self.object_feature_path, vid)
            
            def extract_num(x):
                m = re.findall(r"\d+", x)
                return int(m[-1]) if m else -1

            if osp.isdir(od):
                pkls = sorted([f for f in os.listdir(od) 
                              if f.endswith(".pkl") and not f.startswith("._")],
                             key=extract_num)
                npkl = len(pkls)
                idxs = np.linspace(0, npkl - 1, self.topK_frame).astype(int) if npkl > 0 else []
                
                for i in idxs:
                    try:
                        with open(osp.join(od, pkls[i]), "rb") as fp:
                            cc = pkl.load(fp)
                        
                        if isinstance(cc, dict):
                            feat = cc.get("feat", cc.get("features"))
                            bbox = cc.get("bbox", cc.get("boxes", cc.get("bboxes")))
                            w = cc.get("img_w", 640)
                            h = cc.get("img_h", 480)
                        else:
                            feat, bbox = cc[0], cc[1]
                            w, h = 640, 480
                        
                        if isinstance(feat, np.ndarray):
                            feat = torch.from_numpy(feat)
                        if isinstance(bbox, np.ndarray):
                            bbox = torch.from_numpy(bbox)
                        
                        if feat.shape[0] > self.obj_num:
                            feat = feat[:self.obj_num]
                            bbox = bbox[:self.obj_num]
                        elif feat.shape[0] < self.obj_num:
                            p = self.obj_num - feat.shape[0]
                            feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                            bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
                        
                        bb = torch.from_numpy(transform_bb(bbox.numpy(), w, h)).float()
                        objs.append(torch.cat([feat.float(), bb], -1))
                    except Exception:
                        objs.append(torch.zeros(self.obj_num, 2053))

        # Pad to topK_frame if needed
        while len(objs) < self.topK_frame:
            objs.append(torch.zeros(self.obj_num, 2053))

        return torch.stack(objs)  # [topK_frame, obj_num, 2053]


if __name__ == "__main__":
    print("DataLoader Module Ready (Optimized - Direct PKL Read)")