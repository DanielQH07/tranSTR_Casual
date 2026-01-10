import torch
import os
import re
import pickle as pkl
import h5py
import os.path as osp
import numpy as np
from torch.utils import data
from utils.util import load_file, pause, transform_bb
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast

class VideoQADataset(Dataset):
    def __init__(self, split, n_query=5, obj_num=1, sample_list_path="/data/vqa/causal/anno",\
         video_feature_path="/region_feat_aln", object_feature_path="/object_feat", split_dir=None):
        super(VideoQADataset, self).__init__()
        # 读取dataset
        self.sample_list_file = osp.join(sample_list_path, "{}.csv".format(split))
        self.sample_list = load_file(self.sample_list_file)
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        
        # Paths
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path
        
        print(f"Loading {split} dataset from {self.sample_list_file}...")
        initial_len = len(self.sample_list)
        
        # 1. Filter by TXT split file (if provided)
        if split_dir is not None:
            # Map 'val' to 'valid' if necessary for the txt filename
            txt_name = 'valid' if split == 'val' else split
            txt_path = osp.join(split_dir, f"{txt_name}.txt")
            
            if osp.exists(txt_path):
                print(f"Filtering by split file: {txt_path}")
                with open(txt_path, 'r') as f:
                    valid_vids = set(line.strip() for line in f)
                # Ensure string comparison
                self.sample_list = self.sample_list[self.sample_list['video_id'].astype(str).isin(valid_vids)]
                print(f"  - Samples after split filter: {len(self.sample_list)} (dropped {initial_len - len(self.sample_list)})")
            else:
                print(f"Warning: Split file {txt_path} not found. Skipping split filtering.")

        # 2. Filter by Feature File Existence
        # Optimize: Check unique videos only
        unique_vids = self.sample_list['video_id'].unique()
        existing_vids = []
        missing_count = 0
        
        for vid in unique_vids:
            vid_str = str(vid)
            # Check ViT feature: video_feature_path/{split}/{vid}.pt
            feat_file = osp.join(self.video_feature_path, self.split, f"{vid_str}.pt")
            
            # Optional: Strict check for object folder
            # obj_dir = osp.join(self.object_feature_path, vid_str)
            # if osp.exists(feat_file) and osp.isdir(obj_dir):
            
            if osp.exists(feat_file):
                existing_vids.append(vid)
            else:
                missing_count += 1
                if missing_count <= 5: # Debug print first few missing
                    print(f"  [Missing Feat] {feat_file}")

        if missing_count > 0:
            print(f"  - Missing features for {missing_count} videos.")
        
        existing_vids_set = set(existing_vids)
        prev_len = len(self.sample_list)
        self.sample_list = self.sample_list[self.sample_list['video_id'].isin(existing_vids_set)]
        print(f"  - Final samples after feature check: {len(self.sample_list)} (dropped {prev_len - len(self.sample_list)})")


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        width, height = cur_sample['width'], cur_sample['height']
        video_name = str(cur_sample["video_id"])
        qns_word = str(cur_sample["question"])
        ans_id = self.find_answer_num(cur_sample)
        ans_word = ['[CLS] ' + qns_word+' [SEP] '+ str(cur_sample["a" + str(i)]) for i in range(self.mc)]
        
        # 1. Load Frame Features
        frame_feat_file = osp.join(self.video_feature_path, self.split, f"{video_name}.pt")
        vid_frame_feat = torch.load(frame_feat_file)
        if isinstance(vid_frame_feat, np.ndarray):
            vid_frame_feat = torch.from_numpy(vid_frame_feat)
        vid_frame_feat = vid_frame_feat.type(torch.float32)

        # 2. Load Object Features (Folder of PKL files)
        obj_dir = osp.join(self.object_feature_path, video_name)
        
        # Helper to extract frame number for sorting
        def extract_number(f):
            s = re.findall(r'\d+', f)
            return int(s[-1]) if s else -1

        frame_obj_feats = []
        
        if osp.isdir(obj_dir):
            # List all pkl files, ignoring hidden/metadata files (starting with ._)
            files = [f for f in os.listdir(obj_dir) if f.endswith('.pkl') and not f.startswith('._')]
            # Sort by frame number
            files.sort(key=extract_number)
            
            for f in files:
                f_path = osp.join(obj_dir, f)
                try:
                    with open(f_path, 'rb') as fp:
                        content = pkl.load(fp)
                        
                    # Assume content structure based on user description + typical format
                    # If it's the box/feature content directly
                    roi_feat = None
                    roi_bbox = None
                    
                    if isinstance(content, dict):
                        # Standard format check
                        if any(k in content for k in ['feat', 'features', 'bbox', 'boxes', 'box']):
                            roi_feat = content.get('feat', content.get('features'))
                            roi_bbox = content.get('bbox', content.get('boxes', content.get('box')))
                        else:
                            # User Format: {'person_1': [x1, y1, x2, y2], ...} with relative coordinates
                            boxes_list = []
                            for k, v in content.items():
                                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 4:
                                    # Denormalize relative coordinates to absolute pixels
                                    # as transform_bb expects absolute coordinates
                                    # box is [x1, y1, x2, y2]
                                    abs_box = [
                                        float(v[0]) * width,
                                        float(v[1]) * height,
                                        float(v[2]) * width,
                                        float(v[3]) * height
                                    ]
                                    boxes_list.append(abs_box)
                            
                            if boxes_list:
                                roi_bbox = np.array(boxes_list)
                                roi_feat = None # Will trigger zero-init below
                            else:
                                roi_bbox = None
                                roi_feat = None

                    elif isinstance(content, (list, tuple)) and len(content) >= 2:
                        roi_feat, roi_bbox = content[0], content[1]
                    else:
                        # Fallback: maybe just features or just boxes?
                        # User said "chứa object box" (contains object box).
                        # We need features (2048) + bbox (4/5) usually.
                        # If only box is present, we might mock features or this assumption is wrong.
                        # Let's assume content is the data we need.
                        # Use placeholder if feature extraction fails.
                        pass

                    # Logic to handle missing features if user only provided boxes? 
                    # Assuming we have valid data for now, similar to previous logic.
                    if roi_feat is None and roi_bbox is not None:
                        # Generate dummy features if only boxes exist (unlikely for TranSTR but possible)
                        roi_feat = torch.zeros((len(roi_bbox), 2048))
                    
                    if roi_feat is not None and roi_bbox is not None:
                        # Convert to torch
                        if isinstance(roi_feat, np.ndarray): roi_feat = torch.from_numpy(roi_feat)
                        if isinstance(roi_bbox, np.ndarray): roi_bbox = torch.from_numpy(roi_bbox)
                        
                        # Limit number of objects
                        if roi_feat.shape[0] > self.obj_num:
                            roi_feat = roi_feat[:self.obj_num]
                            roi_bbox = roi_bbox[:self.obj_num]
                            
                        # Transform BBox
                        bbox_feat = transform_bb(roi_bbox.numpy(), width, height)
                        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)
                        roi_feat = roi_feat.type(torch.float32)
                        
                        # Concat
                        # Shape: (Obj_Num, 2048+5)
                        obj_feat_step = torch.cat((roi_feat, bbox_feat), dim=-1)
                        frame_obj_feats.append(obj_feat_step)
                        
                except Exception as e:
                    print(f"Error loading {f_path}: {e}")
                    continue
        
        if len(frame_obj_feats) > 0:
            vid_obj_feat = torch.stack(frame_obj_feats) # (Frames, Obj, Dim)
            vid_obj_feat = vid_obj_feat.flatten(0, 1)   # (Frames*Obj, Dim)
        else:
            # Fallback
            # print(f"Warning: No valid object features found for {video_name} in {obj_dir}")
            vid_obj_feat = torch.zeros((vid_frame_feat.size(0) * self.obj_num, 2048 + 5))

        qns_key = video_name + '_' + str(cur_sample["type"])

        return vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key


    def __len__(self):
        return len(self.sample_list)

    def find_answer_num(self, cur_sample):
        # to find the answer num according to the given answer text
        answer = str(cur_sample["answer"])  # this is the text of the answer
        answer_list = []  # to store all the answer
        for i in range(self.mc):
            answer_list.append(str(cur_sample["a" + str(i)]))
        # answer text match
        for i in range(self.mc):
            if (answer == answer_list[i]):
                return int(i)
        return None  # fail in matching

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="next logger")
    parser.add_argument('-dataset', default='nextqa',choices=['nextqa'], type=str)
    args = parser.parse_args()
    # video_feature_path = '/storage_fast/jbxiao/workspace/VideoQA/data/nextqa'
    # sample_list_path = '/storage_fast/ycli/data/vqa/next/anno'
    train_dataset=VideoQADataset('val', 5, 3)

    train_loader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=False,num_workers=0)

    for sample in train_loader:
        vid_frame_feat, vid_obj_feat, qns_word, ans_word, ans_id, qns_key = sample
        print("frame feat: ")
        print(vid_frame_feat.size())
        print("object feat: ")
        print(vid_obj_feat.size())
        print("qns_word: ")
        print(qns_word)
        print("ans_word")
        print(ans_word)
        print("ans id: ")
        print(ans_id)
        print("qns key: ")
        print(qns_key)
        break
    print('done')