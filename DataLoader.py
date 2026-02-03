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
                 text_feature_path=None, som_feature_path=None):
        """
        Optimized DataLoader with pre-extracted text features.
        
        Args:
            text_feature_path: Path to cached DeBERTa features (optional).
                               If provided, loads from {split}_text_features.pkl
                               If None, text will be processed by model during forward pass
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
        self.som_feature_path = som_feature_path
        
        # Load cached text features if available
        self.text_features = None
        if text_feature_path:
            text_file = osp.join(text_feature_path, f"{split}_text_features.pkl")
            if osp.exists(text_file):
                with open(text_file, 'rb') as f:
                    self.text_features = pkl.load(f)
                if self.verbose:
                    print(f"[{split}] Loaded {len(self.text_features)} cached text features")
            else:
                if self.verbose:
                    print(f"[{split}] WARNING: Text feature file not found: {text_file}")

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

        if not valid_vids and osp.isdir(sample_list_path):
            valid_vids = {d for d in os.listdir(sample_list_path) 
                         if osp.isdir(osp.join(sample_list_path, d))}

        if max_samples and len(valid_vids) > max_samples:
            valid_vids = set(list(valid_vids)[:max_samples])
            if self.verbose:
                print(f"[{split}] Limited to {max_samples} videos")

        # 2. Check feature availability
        vit_available = set()
        obj_available = set()
        
        for vid in valid_vids:
            vit_path = osp.join(self.video_feature_path, f"{vid}.pt")
            if osp.exists(vit_path):
                vit_available.add(vid)
            
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
                            # Include full question for better frame selection context
                            full_question = f"{q['question']} Why?"
                            r = {"video_id": vid, "question": full_question, 
                                 "answer": a["reason"], "type": f"{k}_reason"}
                            for i, c in enumerate(q["reason"]):
                                r[f"a{i}"] = c
                            rows.append(r)
            except Exception as e:
                pass

        self.sample_list = pd.DataFrame(rows)
        
        # Filter by available text features if using cached
        if self.text_features:
            available_keys = set(self.text_features.keys())
            self.sample_list['qns_key'] = self.sample_list.apply(
                lambda x: f"{x['video_id']}_{x['type']}", axis=1)
            before = len(self.sample_list)
            self.sample_list = self.sample_list[self.sample_list['qns_key'].isin(available_keys)]
            if self.verbose and before != len(self.sample_list):
                print(f"[{split}] Filtered to {len(self.sample_list)} samples with text features")
        
        if self.verbose:
            print(f"[{split}] Final: {len(self.sample_list)} QA pairs")

    def _detect_obj_format(self):
        if not osp.exists(self.object_feature_path):
            return 'unknown'
        items = os.listdir(self.object_feature_path)
        for item in items[:5]:
            item_path = osp.join(self.object_feature_path, item)
            if osp.isdir(item_path):
                sub_items = os.listdir(item_path)
                if any(f.endswith('.pkl') for f in sub_items):
                    if 'features_node' in item:
                        return 'kaggle_subdirs'
        return 'per_frame'

    def _has_object_feature(self, vid):
        if self.obj_format == 'kaggle_subdirs':
            for subdir in os.listdir(self.object_feature_path):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    if osp.exists(osp.join(subdir_path, f"{vid}.pkl")):
                        return True
            return False
        else:
            vid_dir = osp.join(self.object_feature_path, vid)
            return osp.isdir(vid_dir) and any(f.endswith('.pkl') for f in os.listdir(vid_dir))

    def _find_object_pkl(self, vid):
        if self.obj_format == 'kaggle_subdirs':
            for subdir in os.listdir(self.object_feature_path):
                subdir_path = osp.join(self.object_feature_path, subdir)
                if osp.isdir(subdir_path):
                    pkl_path = osp.join(subdir_path, f"{vid}.pkl")
                    if osp.exists(pkl_path):
                        return pkl_path
            return None
        return osp.join(self.object_feature_path, vid)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        c = self.sample_list.iloc[idx]
        vid = str(c["video_id"])
        qns = str(c["question"])
        ans_id = int(c["answer"])
        qns_key = f"{vid}_{c['type']}"

        # 1. Load ViT features
        vit_path = osp.join(self.video_feature_path, f"{vid}.pt")
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
        
        # 3. Text features - cached or raw
        if self.text_features and qns_key in self.text_features:
            # Return cached [CLS] features: [5, 768]
            text_feat = torch.from_numpy(self.text_features[qns_key]['cls']).float()
            ans_word = text_feat  # Tensor instead of list of strings
        else:
            # Return raw text for model to process
            ans_word = [f"[CLS] {qns} [SEP] {c[f'a{i}']}" for i in range(self.mc)]

        # 4. Load SoM features
        som_data = self._load_som_features(vid)

        return ff, of, qns, ans_word, ans_id, qns_key, som_data

    def _load_object_features(self, vid):
        objs = []
        
        if self.obj_format == 'kaggle_subdirs':
            pkl_path = self._find_object_pkl(vid)
            if pkl_path and osp.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pkl.load(f)
                    
                    feats = data.get('features')
                    bboxes = data.get('bboxes')
                    
                    if feats is not None and bboxes is not None:
                        if not isinstance(feats, np.ndarray):
                            feats = np.array(feats)
                        if not isinstance(bboxes, np.ndarray):
                            bboxes = np.array(bboxes)
                        
                        num_frames = feats.shape[0]
                        if num_frames > self.topK_frame:
                            indices = np.linspace(0, num_frames - 1, self.topK_frame).astype(int)
                        else:
                            indices = list(range(num_frames))
                        
                        for i in indices:
                            feat = torch.from_numpy(feats[i]).float()
                            bbox = torch.from_numpy(bboxes[i]).float()
                            
                            if feat.shape[0] > self.obj_num:
                                feat = feat[:self.obj_num]
                                bbox = bbox[:self.obj_num]
                            elif feat.shape[0] < self.obj_num:
                                p = self.obj_num - feat.shape[0]
                                feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                                bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
                            
                            bb = torch.from_numpy(transform_bb(bbox.numpy(), 640, 480)).float()
                            objs.append(torch.cat([feat, bb], -1))
                except:
                    pass
        else:
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
                            w, h = cc.get("img_w", 640), cc.get("img_h", 480)
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
                    except:
                        objs.append(torch.zeros(self.obj_num, 2053))

        while len(objs) < self.topK_frame:
            objs.append(torch.zeros(self.obj_num, 2053))

        return torch.stack(objs)

    def _load_som_features(self, vid):
        """
        Load SoM (Set-of-Mark) features from obj_mask_causal directory.
        
        Directory structure:
            obj_mask_causal/
            ├── id_masks/<video_id>.npz   (frame_0..frame_15, each H×W uint8)
            └── metadata_json/<video_id>.json (entity_colors, entity_names, object_to_entity)
        
        Returns:
            dict with keys: 'frame_masks', 'entity_colors', 'entity_names', 'object_to_entity'
            or None if not available
        """
        if not self.som_feature_path:
            return None
        
        mask_path = osp.join(self.som_feature_path, "id_masks", f"{vid}.npz")
        meta_path = osp.join(self.som_feature_path, "metadata_json", f"{vid}.json")
        
        if not osp.exists(mask_path) or not osp.exists(meta_path):
            return None
        
        try:
            # Load masks from npz
            masks_data = np.load(mask_path)
            frame_masks = {}
            for key in masks_data.keys():
                # Actual key format: f0, f1, ..., f15 (not frame_0, frame_1)
                if key.startswith('f') and key[1:].isdigit():
                    frame_idx = int(key[1:])
                    mask = masks_data[key]  # H×W uint8, 0=background, 1..N=entity IDs
                    frame_masks[frame_idx] = torch.from_numpy(mask).long()
            
            # Load metadata from json
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Actual format: {"id_to_label": {"1": "person_1", "2": "person_2"}}
            id_to_label = metadata.get('id_to_label', {})
            
            # Convert to expected format
            entity_names = {int(k): v for k, v in id_to_label.items()}
            entity_colors = {int(k): [255, 0, 0] for k in id_to_label.keys()}  # Placeholder
            object_to_entity = {}  # Not in actual data
            
            return {
                'frame_masks': frame_masks,
                'entity_colors': entity_colors,
                'entity_names': entity_names,
                'object_to_entity': object_to_entity
            }
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Failed to load SoM for {vid}: {e}")
            return None


if __name__ == "__main__":
    print("DataLoader Ready (with cached text feature support)")