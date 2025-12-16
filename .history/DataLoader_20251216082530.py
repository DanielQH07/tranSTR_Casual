import torch
import os
import h5py
import os.path as osp
import numpy as np
import json
import pickle as pkl
from torch.utils import data
from utils.util import load_file, pause, transform_bb, pkload
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast


class VideoQADataset(Dataset):
    """
    DataLoader cho CausalVidQA với output format tương thích NextQA
    
    Output format: (vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key)
    - vid_frame_feat: (T, D) - concatenate appearance + motion features
    - vid_obj_feat: (T*2, obj_num, D) - object features flattened
    - qns_word: question text
    - ans_word: list of "[CLS] question [SEP] answer_i" for each candidate
    - ans_id: ground truth answer index
    - qns_key: video_id + "_" + qtype
    """
        """
        DataLoader for CausalVidQA, single-GPU compatible.
        Output format: (vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key)
        - vid_frame_feat: (T, D) - concatenate appearance + motion features
        - vid_obj_feat: (T, obj_num, D) - object features
        - qns_word: question text
        - ans_word: list of "[CLS] question [SEP] answer_i" for each candidate
        - ans_id: ground truth answer index
        - qns_key: video_id + "_" + qtype
        """
    
    def __init__(self, split, n_query=5, obj_num=1, 
                 sample_list_path=None,  # Path to split pkl files (dataset-split-1)
                 video_feature_path=None,  # Path to visual features
                 text_annotation_path=None,  # Path to text annotations (text.json, answer.json)
                 qtype=-1,  # Question type: -1 for all, 0-5 for specific
                 max_samples=None):  # Limit number of videos for quick testing
        super(VideoQADataset, self).__init__()
        
        self.split = split
        self.mc = n_query  # Number of answer choices (5 for CausalVidQA)
        self.obj_num = obj_num
        self.qtype = qtype
        self.video_feature_path = video_feature_path
        self.text_annotation_path = text_annotation_path
        self.max_samples = max_samples
        
        # Load video ids for this split
        # Handle different naming conventions: val.pkl vs valid.pkl
        split_name = split
        if split == 'val':
            # Try val.pkl first, then valid.pkl
            split_file = osp.join(sample_list_path, 'val.pkl')
            if not osp.exists(split_file):
                split_file = osp.join(sample_list_path, 'valid.pkl')
        else:
            split_file = osp.join(sample_list_path, f'{split}.pkl')
        
        if not osp.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        self.vids = pkload(split_file)
        
        if self.vids is None:
            raise ValueError(f"Failed to load split file: {split_file}")
        
        # Limit number of videos if max_samples is set
        if max_samples is not None and max_samples > 0:
            self.vids = self.vids[:max_samples]
            print(f"Limited to {len(self.vids)} videos (max_samples={max_samples})")
        else:
            print(f"Loaded {len(self.vids)} videos from {split_file}")
        
        # Load video feature index mapping
        idx2vid_file = osp.join(video_feature_path, 'idx2vid.pkl')
        vf_info = pkload(idx2vid_file)
        self.vf_info = dict()
        for idx, vid in enumerate(vf_info):
            if vid in self.vids:
                self.vf_info[vid] = idx
        
        # Load appearance features
        app_file = osp.join(video_feature_path, 'appearance_feat.h5')
        print(f'Loading {app_file}...')
        self.app_feats = dict()
        with h5py.File(app_file, 'r') as fp:
            feats = fp['resnet_features']
            for vid, idx in self.vf_info.items():
                self.app_feats[vid] = feats[idx][...]
        
        # Load motion features
        mot_file = osp.join(video_feature_path, 'motion_feat.h5')
        print(f'Loading {mot_file}...')
        self.mot_feats = dict()
        with h5py.File(mot_file, 'r') as fp:
            feats = fp['resnet_features']
            for vid, idx in self.vf_info.items():
                self.mot_feats[vid] = feats[idx][...]
        
        # Build sample list based on qtype
        self._build_sample_list()

    def _build_sample_list(self):
        """Build list of (video_id, qtype) samples based on qtype setting"""
        self.samples = []
        
        # Question types in CausalVidQA:
        # 0: descriptive
        # 1: explanatory
        # 2: predictive answer
        # 3: predictive reason
        # 4: counterfactual answer
        # 5: counterfactual reason
        
        if self.qtype == -1:
            # All question types
            for vid in self.vids:
                for qt in range(6):
                    self.samples.append((vid, qt))
        elif self.qtype == 0 or self.qtype == 1:
            # Single question type
            for vid in self.vids:
                self.samples.append((vid, self.qtype))
        elif self.qtype == 2:
            # Predictive (answer + reason)
            for vid in self.vids:
                self.samples.append((vid, 2))
                self.samples.append((vid, 3))
        elif self.qtype == 3:
            # Counterfactual (answer + reason)
            for vid in self.vids:
                self.samples.append((vid, 4))
                self.samples.append((vid, 5))
        else:
            for vid in self.vids:
                self.samples.append((vid, self.qtype))
        
        print(f"Total samples: {len(self.samples)}")

    def _load_text(self, vid, qtype):
        """Load question, candidate answers, and ground truth from text annotations"""
        # Try different folder structures
        # Structure 1: text_annotation_path/vid/text.json
        # Structure 2: text_annotation_path/QA/vid/text.json
        text_file = osp.join(self.text_annotation_path, vid, 'text.json')
        answer_file = osp.join(self.text_annotation_path, vid, 'answer.json')
        
        # If not found, try with QA subfolder
        if not osp.exists(text_file):
            text_file = osp.join(self.text_annotation_path, 'QA', vid, 'text.json')
            answer_file = osp.join(self.text_annotation_path, 'QA', vid, 'answer.json')
        
        if not osp.exists(text_file):
            raise FileNotFoundError(f"Text annotation not found for video: {vid}\n"
                                    f"Tried: {osp.join(self.text_annotation_path, vid, 'text.json')}\n"
                                    f"And: {osp.join(self.text_annotation_path, 'QA', vid, 'text.json')}")
        
        with open(text_file, 'r') as f:
            text = json.load(f)
        with open(answer_file, 'r') as f:
            answer = json.load(f)
        
        if qtype == 0:
            qns = text['descriptive']['question']
            cand_ans = text['descriptive']['answer']
            ans_id = answer['descriptive']['answer']
        elif qtype == 1:
            qns = text['explanatory']['question']
            cand_ans = text['explanatory']['answer']
            ans_id = answer['explanatory']['answer']
        elif qtype == 2:
            qns = text['predictive']['question']
            cand_ans = text['predictive']['answer']
            ans_id = answer['predictive']['answer']
        elif qtype == 3:
            qns = text['predictive']['question']
            cand_ans = text['predictive']['reason']
            ans_id = answer['predictive']['reason']
        elif qtype == 4:
            qns = text['counterfactual']['question']
            cand_ans = text['counterfactual']['answer']
            ans_id = answer['counterfactual']['answer']
        elif qtype == 5:
            qns = text['counterfactual']['question']
            cand_ans = text['counterfactual']['reason']
            ans_id = answer['counterfactual']['reason']
        else:
            raise ValueError(f"Invalid qtype: {qtype}")
        
        return qns, cand_ans, ans_id


    def __getitem__(self, idx):
        vid, qtype = self.samples[idx]
        
        # Load text data
        qns_word, cand_ans, ans_id = self._load_text(vid, qtype)
        
        # Format answer words like NextQA: "[CLS] question [SEP] answer"
        ans_word = ['[CLS] ' + qns_word + ' [SEP] ' + str(cand_ans[i]) for i in range(self.mc)]
        
        # Load video features
        app_feat = self.app_feats[vid]
        mot_feat = self.mot_feats[vid]
        
        # Handle different feature shapes
        # app_feat could be (T, D) or (T, N, D) where N is number of clips
        # mot_feat could be (T, D) or (T, N, D)
        
        # Squeeze or reshape if needed to get (T, D)
        if app_feat.ndim == 3:
            # Shape is (T, N, D) - take mean over N or reshape
            app_feat = app_feat.mean(axis=1) if app_feat.shape[1] > 1 else app_feat.squeeze(1)
        if mot_feat.ndim == 3:
            mot_feat = mot_feat.mean(axis=1) if mot_feat.shape[1] > 1 else mot_feat.squeeze(1)
        
        # Ensure both have same shape
        if app_feat.ndim == 1:
            app_feat = app_feat[np.newaxis, :]  # (1, D)
        if mot_feat.ndim == 1:
            mot_feat = mot_feat[np.newaxis, :]  # (1, D)
        
        # Frame feature: concatenate app + mot -> (T, D*2)
        frame_feat = np.concatenate([app_feat, mot_feat], axis=-1)
        vid_frame_feat = torch.from_numpy(frame_feat).type(torch.float32)
        
        # Object features - CausalVidQA không có region features riêng
        # Ta dùng appearance features làm "object" features với dummy bbox
        # Shape expected by model: (F, O, 2053) where F=frames, O=objects
        T = app_feat.shape[0]
        D_obj = app_feat.shape[-1]  # 2048
        
        # Tạo object feature: (T, obj_num, 2048)
        # Replicate frame feature cho mỗi object slot
        obj_feat = np.tile(app_feat[:, np.newaxis, :], (1, self.obj_num, 1))  # (T, obj_num, 2048)
        
        # Add dummy bbox features (5 dims: x1, y1, x2, y2, area normalized)
        # Shape: (T, obj_num, 5)
        dummy_bbox = np.zeros((T, self.obj_num, 5), dtype=np.float32)
        # Set some default values for bbox (normalized coordinates)
        dummy_bbox[:, :, :4] = np.array([0.0, 0.0, 1.0, 1.0])  # full frame
        dummy_bbox[:, :, 4] = 1.0  # area = 1 (full frame)
        
        # Concatenate: (T, obj_num, 2048+5=2053)
        obj_feat = np.concatenate([obj_feat, dummy_bbox], axis=-1)
        vid_obj_feat = torch.from_numpy(obj_feat).type(torch.float32)
        
        # Question key format
        qns_key = vid + '_' + str(qtype)
        
        return vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CausalVidQA DataLoader Test")
    parser.add_argument('--split', default='val', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--sample_list_path', type=str, required=True,
                        help='Path to split pkl files (dataset-split-1)')
    parser.add_argument('--video_feature_path', type=str, required=True,
                        help='Path to visual features (appearance_feat.h5, motion_feat.h5)')
    parser.add_argument('--text_annotation_path', type=str, required=True,
                        help='Path to text annotations (folders with text.json, answer.json)')
    parser.add_argument('--qtype', type=int, default=-1,
                        help='Question type: -1 for all, 0-5 for specific')
    parser.add_argument('--n_query', type=int, default=5, help='Number of answer choices')
    parser.add_argument('--obj_num', type=int, default=1, help='Number of objects')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    
    # Example with kagglehub:
    # import kagglehub
    # args.sample_list_path = kagglehub.dataset_download('lusnaw/dataset-split-1')
    # args.video_feature_path = kagglehub.dataset_download('lusnaw/visual-feature')
    # args.text_annotation_path = kagglehub.dataset_download('lusnaw/text-annotation')
    
    train_dataset = VideoQADataset(
        split=args.split,
        n_query=args.n_query,
        obj_num=args.obj_num,
        sample_list_path=args.sample_list_path,
        video_feature_path=args.video_feature_path,
        text_annotation_path=args.text_annotation_path,
        qtype=args.qtype
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    for sample in train_loader:
        vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key = sample
        print("=" * 50)
        print("Frame feat shape:", vid_frame_feat.size())
        print("Object feat shape:", vid_obj_feat.size())
        print("Question:", qns_word)
        print("Answer choices:", ans_word)
        print("Answer ID:", ans_id)
        print("Question key:", qns_key)
        print("=" * 50)
        break
    
    print('Done!')