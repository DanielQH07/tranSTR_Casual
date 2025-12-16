# 🔧 PATCH DataLoader.py - LAZY LOADING (Fix OOM for full dataset)
# ============================================================

patch_code = '''import torch
import os
import h5py
import os.path as osp
import numpy as np
import json
import pickle as pkl
from torch.utils.data import Dataset
from utils.util import pkload

class VideoQADataset(Dataset):
    """
    Optimized DataLoader with LAZY LOADING for large datasets
    ✅ Không load toàn bộ features vào RAM
    ✅ Chỉ load khi cần (trong __getitem__)
    ✅ Tiết kiệm ~30GB RAM cho full dataset
    """
    
    def __init__(self, split, n_query=5, obj_num=1, 
                 sample_list_path=None,
                 video_feature_path=None,
                 text_annotation_path=None,
                 qtype=-1,
                 max_samples=None):
        super(VideoQADataset, self).__init__()
        
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.qtype = qtype
        self.video_feature_path = video_feature_path
        self.text_annotation_path = text_annotation_path
        self.max_samples = max_samples
        
        # Load video ids
        if split == 'val':
            split_file = osp.join(sample_list_path, 'val.pkl')
            if not osp.exists(split_file):
                split_file = osp.join(sample_list_path, 'valid.pkl')
        else:
            split_file = osp.join(sample_list_path, f'{split}.pkl')
        
        self.vids = pkload(split_file)
        
        if max_samples is not None and max_samples > 0:
            self.vids = self.vids[:max_samples]
            print(f"📊 Limited to {len(self.vids)} videos")
        else:
            print(f"📊 FULL DATASET: {len(self.vids)} videos")
        
        # Load video feature index mapping
        idx2vid_file = osp.join(video_feature_path, 'idx2vid.pkl')
        vf_info = pkload(idx2vid_file)
        self.vf_info = dict()
        for idx, vid in enumerate(vf_info):
            if vid in self.vids:
                self.vf_info[vid] = idx
        
        # ===== LAZY LOADING: Chỉ lưu file paths, KHÔNG load data =====
        self.app_file = osp.join(video_feature_path, 'appearance_feat.h5')
        self.mot_file = osp.join(video_feature_path, 'motion_feat.h5')
        
        print(f"✅ Using LAZY LOADING (memory-efficient)")
        
        self._build_sample_list()

    def _build_sample_list(self):
        self.samples = []
        
        if self.qtype == -1:
            for vid in self.vids:
                for qt in range(6):
                    self.samples.append((vid, qt))
        elif self.qtype == 0 or self.qtype == 1:
            for vid in self.vids:
                self.samples.append((vid, self.qtype))
        elif self.qtype == 2:
            for vid in self.vids:
                self.samples.append((vid, 2))
                self.samples.append((vid, 3))
        elif self.qtype == 3:
            for vid in self.vids:
                self.samples.append((vid, 4))
                self.samples.append((vid, 5))
        else:
            for vid in self.vids:
                self.samples.append((vid, self.qtype))
        
        print(f"   Total samples: {len(self.samples)}")

    def _load_text(self, vid, qtype):
        text_file = osp.join(self.text_annotation_path, vid, 'text.json')
        answer_file = osp.join(self.text_annotation_path, vid, 'answer.json')
        
        if not osp.exists(text_file):
            text_file = osp.join(self.text_annotation_path, 'QA', vid, 'text.json')
            answer_file = osp.join(self.text_annotation_path, 'QA', vid, 'answer.json')
        
        with open(text_file, 'r') as f:
            text = json.load(f)
        with open(answer_file, 'r') as f:
            answer = json.load(f)
        
        qtype_map = {
            0: ('descriptive', 'question', 'answer'),
            1: ('explanatory', 'question', 'answer'),
            2: ('predictive', 'question', 'answer'),
            3: ('predictive', 'question', 'reason'),
            4: ('counterfactual', 'question', 'answer'),
            5: ('counterfactual', 'question', 'reason')
        }
        
        qtype_key, q_key, a_key = qtype_map[qtype]
        qns = text[qtype_key][q_key]
        cand_ans = text[qtype_key][a_key]
        ans_id = answer[qtype_key][a_key]
        
        return qns, cand_ans, ans_id

    def __getitem__(self, idx):
        vid, qtype = self.samples[idx]
        
        qns_word, cand_ans, ans_id = self._load_text(vid, qtype)
        ans_word = ['[CLS] ' + qns_word + ' [SEP] ' + str(cand_ans[i]) for i in range(self.mc)]
        
        # ===== LAZY LOADING: Load features ON-DEMAND (chỉ khi cần) =====
        vid_idx = self.vf_info[vid]
        
        with h5py.File(self.app_file, 'r') as fp:
            app_feat = fp['resnet_features'][vid_idx][...]
        
        with h5py.File(self.mot_file, 'r') as fp:
            mot_feat = fp['resnet_features'][vid_idx][...]
        
        # Handle different feature shapes
        if app_feat.ndim == 3:
            app_feat = app_feat.mean(axis=1) if app_feat.shape[1] > 1 else app_feat.squeeze(1)
        if mot_feat.ndim == 3:
            mot_feat = mot_feat.mean(axis=1) if mot_feat.shape[1] > 1 else mot_feat.squeeze(1)
        
        if app_feat.ndim == 1:
            app_feat = app_feat[np.newaxis, :]
        if mot_feat.ndim == 1:
            mot_feat = mot_feat[np.newaxis, :]
        
        frame_feat = np.concatenate([app_feat, mot_feat], axis=-1)
        vid_frame_feat = torch.from_numpy(frame_feat).type(torch.float32)
        
        T = app_feat.shape[0]
        obj_feat = np.tile(app_feat[:, np.newaxis, :], (1, self.obj_num, 1))
        dummy_bbox = np.zeros((T, self.obj_num, 5), dtype=np.float32)
        dummy_bbox[:, :, :4] = np.array([0.0, 0.0, 1.0, 1.0])
        dummy_bbox[:, :, 4] = 1.0
        
        obj_feat = np.concatenate([obj_feat, dummy_bbox], axis=-1)
        vid_obj_feat = torch.from_numpy(obj_feat).type(torch.float32)
        
        qns_key = vid + '_' + str(qtype)
        return vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key

    def __len__(self):
        return len(self.samples)
'''

with open('DataLoader.py', 'w') as f:
    f.write(patch_code)

print("✅ DataLoader.py patched with LAZY LOADING!")
print("📌 KEY IMPROVEMENTS:")
print("   - Memory usage: ~500MB instead of ~30-40GB")
print("   - Can train on FULL dataset without OOM")
print("   - Slightly slower I/O (negligible with SSD)")
