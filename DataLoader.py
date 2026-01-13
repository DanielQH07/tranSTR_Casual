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
        Args:
            split: 'train', 'val', or 'test'
            n_query: number of answer choices (5 for multiple choice)
            obj_num: number of objects per frame
            sample_list_path: path to annotation directory (contains {video_id}/text.json, answer.json)
            video_feature_path: path to ViT features (contains {split}/{video_id}.pt)
            object_feature_path: path to object features (contains {video_id}/*.pkl)
            split_dir: path to split files (train.pkl, valid.pkl, test.pkl)
            topK_frame: number of frames to sample
            max_samples: maximum number of samples to load (None = no limit)
            verbose: whether to print detailed logs
        """
        super().__init__()
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        self.topK_frame = topK_frame
        self.verbose = verbose

        # 1. Load Split Video IDs from PKL file
        valid_vids = set()
        if split_dir:
            # Map 'val' to 'valid' for file naming
            pkl_name = 'valid' if split == 'val' else split
            pkl_path = osp.join(split_dir, f"{pkl_name}.pkl")
            txt_path = osp.join(split_dir, f"{pkl_name}.txt")
            
            if osp.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pkl.load(f)
                # Handle different pkl structures
                if isinstance(data, (list, set)):
                    valid_vids = set(data)
                elif isinstance(data, dict):
                    valid_vids = set(data.keys())
                else:
                    valid_vids = set(data)
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {pkl_path}")
            elif osp.exists(txt_path):
                # Fallback to txt file
                with open(txt_path) as f:
                    valid_vids = {l.strip() for l in f if l.strip()}
                if self.verbose:
                    print(f"[{split}] Loaded {len(valid_vids)} video IDs from {txt_path}")
            else:
                if self.verbose:
                    print(f"[{split}] Warning: No split file found at {pkl_path} or {txt_path}")

        # Fallback: use all directories in sample_list_path
        if not valid_vids and osp.isdir(sample_list_path):
            valid_vids = {d for d in os.listdir(sample_list_path) 
                         if osp.isdir(osp.join(sample_list_path, d))}
            if self.verbose:
                print(f"[{split}] Fallback: Using {len(valid_vids)} video dirs from {sample_list_path}")

        # Apply max_samples limit for train split
        if max_samples is not None and split == 'train' and len(valid_vids) > max_samples:
            valid_vids = set(list(valid_vids)[:max_samples])
            if self.verbose:
                print(f"[{split}] Limited to {max_samples} videos")

        # 2. Check feature availability BEFORE parsing annotations
        vit_available = set()
        obj_available = set()
        
        if self.verbose:
            print(f"[{split}] Checking feature availability...")
        
        for vid in valid_vids:
            # Check ViT features
            vit_path = osp.join(self.video_feature_path, self.split, f"{vid}.pt")
            if osp.exists(vit_path):
                vit_available.add(vid)
            
            # Check Object features
            obj_dir = osp.join(self.object_feature_path, vid)
            if osp.isdir(obj_dir):
                pkl_files = [f for f in os.listdir(obj_dir) 
                            if f.endswith('.pkl') and not f.startswith('._')]
                if pkl_files:
                    obj_available.add(vid)
        
        # Find videos with BOTH features
        both_available = vit_available & obj_available
        vit_only = vit_available - obj_available
        obj_only = obj_available - vit_available
        neither = valid_vids - vit_available - obj_available
        
        if self.verbose:
            print(f"[{split}] Feature Check Results:")
            print(f"  - Total video IDs from split: {len(valid_vids)}")
            print(f"  - ViT features found: {len(vit_available)}")
            print(f"  - Object features found: {len(obj_available)}")
            print(f"  - Both features available: {len(both_available)}")
            if vit_only:
                print(f"  - ViT only (missing object): {len(vit_only)}")
                if len(vit_only) <= 5:
                    print(f"    Examples: {list(vit_only)[:5]}")
            if obj_only:
                print(f"  - Object only (missing ViT): {len(obj_only)}")
                if len(obj_only) <= 5:
                    print(f"    Examples: {list(obj_only)[:5]}")
            if neither:
                print(f"  - Neither feature found: {len(neither)}")
                if len(neither) <= 5:
                    print(f"    Examples: {list(neither)[:5]}")

        # Only use videos with both features
        valid_vids = both_available

        # 3. Parse JSON Annotations
        rows = []
        annotation_errors = []
        
        for vid in valid_vids:
            vp = osp.join(sample_list_path, vid)
            tj, aj = osp.join(vp, "text.json"), osp.join(vp, "answer.json")
            
            if not (osp.exists(tj) and osp.exists(aj)):
                annotation_errors.append(f"{vid}: missing json files")
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
                            
                        # Handle reason questions for predictive/counterfactual
                        if k in ["predictive", "counterfactual"] and "reason" in q and "reason" in a:
                            r = {"video_id": vid, "question": "Why?", 
                                 "answer": a["reason"], "type": f"{k}_reason"}
                            for i, c in enumerate(q["reason"]):
                                r[f"a{i}"] = c
                            rows.append(r)
            except Exception as e:
                annotation_errors.append(f"{vid}: {str(e)}")

        if self.verbose and annotation_errors:
            print(f"[{split}] Annotation parsing errors: {len(annotation_errors)}")
            for err in annotation_errors[:5]:
                print(f"    {err}")
            if len(annotation_errors) > 5:
                print(f"    ... and {len(annotation_errors) - 5} more")

        self.sample_list = pd.DataFrame(rows)
        
        if self.verbose:
            print(f"[{split}] Final dataset: {len(self.sample_list)} QA pairs from {len(valid_vids)} videos")
            if len(self.sample_list) > 0:
                # Show question type distribution
                type_counts = self.sample_list['type'].value_counts()
                print(f"[{split}] Question type distribution:")
                for qtype, count in type_counts.items():
                    print(f"    {qtype}: {count}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        c = self.sample_list.iloc[idx]
        vid = str(c["video_id"])
        qns = str(c["question"])
        ans_id = int(c["answer"])
        ans_word = [f"[CLS] {qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]

        # 1. Frame features - SAMPLE/PAD to topK_frame
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

        # 2. Object features - [topK_frame, obj_num, 2053]
        od = osp.join(self.object_feature_path, vid)
        objs = []
        
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
                        bbox = cc.get("bbox", cc.get("boxes", cc.get("box")))
                        w = cc.get("img_w", 640)
                        h = cc.get("img_h", 480)
                    else:
                        feat, bbox = cc[0], cc[1]
                        w, h = 640, 480
                    
                    if isinstance(feat, np.ndarray):
                        feat = torch.from_numpy(feat)
                    if isinstance(bbox, np.ndarray):
                        bbox = torch.from_numpy(bbox)
                    
                    # Limit/pad objects
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

        of = torch.stack(objs)  # [topK_frame, obj_num, 2053]
        
        qns_key = f"{vid}_{c['type']}"
        return ff, of, qns, ans_word, ans_id, qns_key


def verify_dataset(split, args, max_check=10):
    """
    Utility function to verify dataset configuration and sample some data.
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING {split.upper()} DATASET")
    print(f"{'='*60}")
    
    dataset = VideoQADataset(
        split=split,
        n_query=args.n_query,
        obj_num=args.objs,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_root,
        object_feature_path=args.object_feature_path,
        split_dir=args.split_dir_txt,
        topK_frame=args.topK_frame,
        max_samples=getattr(args, 'max_train_samples', None) if split == 'train' else None,
        verbose=True
    )
    
    if len(dataset) == 0:
        print("WARNING: Dataset is empty!")
        return dataset
    
    # Sample check
    print(f"\n[{split}] Sampling {min(max_check, len(dataset))} items for verification...")
    
    for i in range(min(max_check, len(dataset))):
        try:
            ff, of, qns, ans_word, ans_id, qns_key = dataset[i]
            print(f"  [{i}] {qns_key}")
            print(f"      ViT shape: {ff.shape}, Object shape: {of.shape}")
            print(f"      Question: {qns[:50]}...")
            print(f"      Answer ID: {ans_id}")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
    
    print(f"{'='*60}\n")
    return dataset


if __name__ == "__main__":
    print("DataLoader Module Ready")
    print("Use verify_dataset(split, args) to check your configuration")