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
        # 1. Load / Discover Video IDs
        self.split = split
        self.mc = n_query
        self.obj_num = obj_num
        self.video_feature_path = video_feature_path
        self.object_feature_path = object_feature_path

        # Determine valid video IDs based on Split Text Files
        valid_vids = []
        if split_dir is not None:
            txt_name = 'valid' if split == 'val' else split
            txt_path = osp.join(split_dir, f"{txt_name}.txt")
            if osp.exists(txt_path):
                print(f"Loading split from: {txt_path}")
                with open(txt_path, 'r') as f:
                    valid_vids = [line.strip() for line in f if line.strip()]
            else:
                print(f"Warning: Split file {txt_path} not found.")

        # 2. Parse JSON Annotations from folders
        # Expected structure: sample_list_path / video_id / {text.json, answer.json}
        data_rows = []
        
        # If no split file provided, try to list directories in sample_list_path
        if not valid_vids and os.path.isdir(sample_list_path):
            valid_vids = [d for d in os.listdir(sample_list_path) if os.path.isdir(os.path.join(sample_list_path, d))]

        print(f"Parsing annotations for {len(valid_vids)} videos...")
        
        for vid in valid_vids:
            vid_path = osp.join(sample_list_path, vid)
            text_json_path = osp.join(vid_path, 'text.json')
            ans_json_path = osp.join(vid_path, 'answer.json')

            if not (osp.exists(text_json_path) and osp.exists(ans_json_path)):
                # print(f"Missing annotation files for {vid}")
                continue

            try:
                # Load JSONs
                with open(text_json_path, 'r', encoding='utf-8') as f:
                    text_data = json.load(f)
                with open(ans_json_path, 'r', encoding='utf-8') as f:
                    ans_data = json.load(f)

                # Helper to add row
                def add_row(q_type, q_text, candidates, ans_idx, sub_type=None):
                    if not candidates or ans_idx is None: return
                    row = {
                        'video_id': vid,
                        'question': q_text,
                        'answer': ans_idx, # Index 0-4
                        'type': q_type if sub_type is None else f"{q_type}_{sub_type}",
                        'width': 640, # Default/Dummy width
                        'height': 480 # Default/Dummy height
                    }
                    # Add candidate columns a0..a4
                    for i, cand in enumerate(candidates):
                        row[f"a{i}"] = cand
                    data_rows.append(row)

                # Flatten the structure
                # Root keys: descriptive, explanatory, predictive, counterfactual
                for key in ['descriptive', 'explanatory', 'predictive', 'counterfactual']:
                    if key in text_data and key in ans_data:
                        t_item = text_data[key]
                        a_item = ans_data[key]
                        
                        # 1. Main Question
                        # text: "question", "answer" (list)
                        # answer: "answer" (int index)
                        if 'question' in t_item and 'answer' in t_item and 'answer' in a_item:
                            add_row(key, t_item['question'], t_item['answer'], a_item['answer'])
                        
                        # 2. Reasoning (for predictive/counterfactual)
                        # User's JSON shows "reason" list in text and "reason" int in answer
                        # But NO check for separate question text. 
                        # We will skip creating a separate sample for reason to avoid duplicating the main question
                        # unless generic "Why?" is appended, but that is risky without specific data.
                        # Implementation: Only main Q&A pairs loaded for now.

            except Exception as e:
                print(f"Error parsing {vid}: {e}")

        # Create DataFrame
        self.sample_list = pd.DataFrame(data_rows)
        print(f"Loaded {len(self.sample_list)} questions from {len(valid_vids)} videos.")

        # 3. Filter by existing features (Logic preserved)
        initial_len = len(self.sample_list)
        if initial_len == 0:
            print("Warning: No samples loaded!")
            return

        unique_vids = self.sample_list['video_id'].unique()
        existing_vids = set()
        
        for vid in unique_vids:
            vid_str = str(vid)
            feat_file = osp.join(self.video_feature_path, self.split, f"{vid_str}.pt")
            if osp.exists(feat_file):
                existing_vids.add(vid)
        
        self.sample_list = self.sample_list[self.sample_list['video_id'].isin(existing_vids)]
        print(f"Final samples after feature check: {len(self.sample_list)} (dropped {initial_len - len(self.sample_list)})")


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
        # We already stored the integer answer index in the dataframe during parsing
        return int(cur_sample["answer"])
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