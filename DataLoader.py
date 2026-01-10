import torch
import os
import re
import json
import pandas as pd
import pickle as pkl
import h5py
import os.path as osp
import numpy as np
from torch.utils import data
from utils.util import load_file, pause, transform_bb
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast

class VideoQADataset(Dataset):
    def __init__(self, split, n_query=5, obj_num=10, sample_list_path="/data/vqa/causal/anno",\
         video_feature_path="/region_feat_aln", object_feature_path="/object_feat", split_dir=None, topK_frame=16):
        super(VideoQADataset, self).__init__()
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        self.topK_frame = topK_frame

        # 1. Load Split Video IDs
        valid_vids = set()
        if split_dir is not None:
            txt_name = 'valid' if split == 'val' else split
            txt_path = osp.join(split_dir, f"{txt_name}.txt")
            if osp.exists(txt_path):
                print(f"Loading split from: {txt_path}")
                with open(txt_path, 'r') as f:
                    valid_vids = {line.strip() for line in f if line.strip()}
            else:
                print(f"Warning: Split file {txt_path} not found.")

        # 2. Parse JSON Annotations
        data_rows = []
        
        if not valid_vids and os.path.isdir(sample_list_path):
             valid_vids = {d for d in os.listdir(sample_list_path) if os.path.isdir(os.path.join(sample_list_path, d))}

        print(f"Scanning annotations for {len(valid_vids)} videos...")
        for vid in valid_vids:
            vid_path = osp.join(sample_list_path, vid)
            text_json = osp.join(vid_path, 'text.json')
            ans_json = osp.join(vid_path, 'answer.json')
            
            if not(osp.exists(text_json) and osp.exists(ans_json)): continue
            
            try:
                with open(text_json, 'r', encoding='utf-8') as f: t_data = json.load(f)
                with open(ans_json, 'r', encoding='utf-8') as f: a_data = json.load(f)
                
                for key in ['descriptive', 'explanatory', 'predictive', 'counterfactual']:
                    if key in t_data and key in a_data:
                        q_obj = t_data[key]
                        a_obj = a_data[key]
                        
                        if 'question' in q_obj and 'answer' in q_obj and 'answer' in a_obj:
                            row = {
                                'video_id': vid,
                                'question': q_obj['question'],
                                'answer': a_obj['answer'],
                                'type': key,
                                'width': 640, 'height': 480 
                            }
                            for i, cand in enumerate(q_obj['answer']):
                                row[f"a{i}"] = cand
                            data_rows.append(row)
                        
                        if key in ['predictive', 'counterfactual'] and 'reason' in q_obj and 'reason' in a_obj:
                             row_reason = {
                                'video_id': vid,
                                'question': "Why?",
                                'answer': a_obj['reason'],
                                'type': f"{key}_reason",
                                'width': 640, 'height': 480 
                            }
                             for i, cand in enumerate(q_obj['reason']):
                                row_reason[f"a{i}"] = cand
                             data_rows.append(row_reason)

            except Exception as e:
                print(f"Error parsing {vid}: {e}")

        self.sample_list = pd.DataFrame(data_rows)
        print(f"Loaded {len(self.sample_list)} QA pairs.")

        # 3. Filter by Video Feature Existence
        initial_len = len(self.sample_list)
        if initial_len > 0:
            existing_vids = set()
            unique_vids = self.sample_list['video_id'].unique()
            for vid in unique_vids:
                vid_path = osp.join(self.video_feature_path, self.split, f"{vid}.pt")
                if osp.exists(vid_path): 
                    existing_vids.add(vid)
                    
            self.sample_list = self.sample_list[self.sample_list['video_id'].isin(existing_vids)]
            print(f"Final dataset size: {len(self.sample_list)} (Dropped {initial_len - len(self.sample_list)})")


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        video_name = str(cur_sample["video_id"])
        qns_word = str(cur_sample["question"])
        ans_id = int(cur_sample["answer"])
        ans_word = ['[CLS] ' + qns_word+' [SEP] '+ str(cur_sample[f"a{i}"]) for i in range(self.mc)]
        
        # ===== 1. Load Video (Frame) Features =====
        frame_feat_path = osp.join(self.video_feature_path, self.split, f"{video_name}.pt")
        vid_frame_feat = torch.load(frame_feat_path)
        if isinstance(vid_frame_feat, np.ndarray): vid_frame_feat = torch.from_numpy(vid_frame_feat)
        vid_frame_feat = vid_frame_feat.type(torch.float32)
        
        # ===== CRITICAL: Sample/Pad frame features to match topK_frame =====
        num_frames = vid_frame_feat.shape[0]
        if num_frames > self.topK_frame:
            # Uniform sampling
            indices = np.linspace(0, num_frames - 1, self.topK_frame).astype(int)
            vid_frame_feat = vid_frame_feat[indices]
        elif num_frames < self.topK_frame:
            # Pad with zeros
            pad_size = self.topK_frame - num_frames
            pad_feat = torch.zeros((pad_size, vid_frame_feat.shape[1]), dtype=vid_frame_feat.dtype)
            vid_frame_feat = torch.cat([vid_frame_feat, pad_feat], dim=0)
        # Now vid_frame_feat is [topK_frame, feat_dim]
        
        # ===== 2. Load Object Features (PKL Folder) =====
        obj_dir = osp.join(self.object_feature_path, video_name)
        frame_obj_feats = []
        
        def extract_number(f):
             s = re.findall(r'\d+', f)
             return int(s[-1]) if s else -1

        if osp.isdir(obj_dir):
            pkl_files = [f for f in os.listdir(obj_dir) if f.endswith('.pkl') and not f.startswith('._')]
            pkl_files.sort(key=extract_number)
            
            num_files = len(pkl_files)
            target_frames = self.topK_frame
            indices = np.linspace(0, num_files - 1, target_frames).astype(int) if num_files > 0 else []
            
            for i in indices:
                f_path = osp.join(obj_dir, pkl_files[i])
                try:
                    with open(f_path, 'rb') as fp: content = pkl.load(fp)
                    
                    roi_feat, roi_bbox, w, h = None, None, 640, 480
                    
                    if isinstance(content, dict):
                        roi_feat = content.get('feat', content.get('features'))
                        roi_bbox = content.get('bbox', content.get('boxes', content.get('box')))
                        w = content.get('img_w', content.get('width', 640))
                        h = content.get('img_h', content.get('height', 480))
                    elif isinstance(content, (list, tuple)) and len(content) >= 4:
                        roi_feat, roi_bbox, w, h = content[0], content[1], content[2], content[3]
                    
                    if roi_feat is not None and roi_bbox is not None:
                        if isinstance(roi_feat, np.ndarray): roi_feat = torch.from_numpy(roi_feat)
                        if isinstance(roi_bbox, np.ndarray): roi_bbox = torch.from_numpy(roi_bbox)
                        
                        if roi_feat.shape[0] > self.obj_num:
                            roi_feat = roi_feat[:self.obj_num]
                            roi_bbox = roi_bbox[:self.obj_num]
                        elif roi_feat.shape[0] < self.obj_num:
                             pad_num = self.obj_num - roi_feat.shape[0]
                             feat_pad = torch.zeros((pad_num, roi_feat.shape[1]), dtype=roi_feat.dtype)
                             bbox_pad = torch.zeros((pad_num, roi_bbox.shape[1]), dtype=roi_bbox.dtype)
                             roi_feat = torch.cat((roi_feat, feat_pad), dim=0)
                             roi_bbox = torch.cat((roi_bbox, bbox_pad), dim=0)

                        bbox_feat = transform_bb(roi_bbox.numpy(), w, h)
                        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)
                        roi_feat = roi_feat.type(torch.float32)
                        
                        obj_feat_step = torch.cat((roi_feat, bbox_feat), dim=-1)
                        frame_obj_feats.append(obj_feat_step)
                    else:
                        frame_obj_feats.append(torch.zeros((self.obj_num, 2048+5)))
                        
                except Exception as e:
                    frame_obj_feats.append(torch.zeros((self.obj_num, 2048+5)))
        
        # Ensure exact length
        if len(frame_obj_feats) == 0:
             frame_obj_feats = [torch.zeros((self.obj_num, 2048+5)) for _ in range(self.topK_frame)]
        elif len(frame_obj_feats) < self.topK_frame:
            while len(frame_obj_feats) < self.topK_frame:
                frame_obj_feats.append(torch.zeros((self.obj_num, 2048+5)))

        vid_obj_feat = torch.stack(frame_obj_feats) # [topK_frame, obj_num, 2053]

        qns_key = video_name + '_' + str(cur_sample["type"])
        return vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key

    def __len__(self):
        return len(self.sample_list)

    def find_answer_num(self, cur_sample):
        return int(cur_sample["answer"])

if __name__ == "__main__":
    print("DataLoader Module Ready")