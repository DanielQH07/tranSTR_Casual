"""
Script để kiểm tra cấu trúc và format dữ liệu CausalVidQA
Chạy script này để verify data trước khi train

Usage:
    python check_data.py \
        --sample_list_path /path/to/dataset-split-1 \
        --video_feature_path /path/to/visual-feature \
        --text_annotation_path /path/to/text-annotation
"""

import os
import os.path as osp
import argparse
import pickle
import json
import h5py
import numpy as np


def check_split_files(sample_list_path):
    """Kiểm tra split files (train.pkl, valid.pkl, test.pkl)"""
    print("\n" + "="*60)
    print("1. CHECKING SPLIT FILES")
    print("="*60)
    
    split_files = ['train.pkl', 'valid.pkl', 'val.pkl', 'test.pkl']
    found_splits = {}
    
    for split_file in split_files:
        path = osp.join(sample_list_path, split_file)
        if osp.exists(path):
            with open(path, 'rb') as f:
                vids = pickle.load(f)
            found_splits[split_file] = vids
            print(f"  ✓ {split_file}: {len(vids)} videos")
            if len(vids) <= 5:
                print(f"    Video IDs: {vids}")
            else:
                print(f"    First 3: {vids[:3]}")
                print(f"    Last 3: {vids[-3:]}")
        else:
            print(f"  ✗ {split_file}: NOT FOUND")
    
    # Check for issues
    for name, vids in found_splits.items():
        if len(vids) < 100:
            print(f"\n  ⚠️  WARNING: {name} chỉ có {len(vids)} videos! Có thể bị thiếu data.")
    
    return found_splits


def check_visual_features(video_feature_path, sample_vids=None):
    """Kiểm tra visual features (appearance, motion)"""
    print("\n" + "="*60)
    print("2. CHECKING VISUAL FEATURES")
    print("="*60)
    
    # Check idx2vid.pkl
    idx2vid_path = osp.join(video_feature_path, 'idx2vid.pkl')
    if osp.exists(idx2vid_path):
        with open(idx2vid_path, 'rb') as f:
            idx2vid = pickle.load(f)
        print(f"  ✓ idx2vid.pkl: {len(idx2vid)} videos indexed")
        print(f"    First 3: {idx2vid[:3]}")
    else:
        print(f"  ✗ idx2vid.pkl: NOT FOUND")
        idx2vid = None
    
    # Check appearance features
    app_path = osp.join(video_feature_path, 'appearance_feat.h5')
    if osp.exists(app_path):
        with h5py.File(app_path, 'r') as f:
            keys = list(f.keys())
            print(f"  ✓ appearance_feat.h5")
            print(f"    Keys: {keys}")
            for key in keys:
                shape = f[key].shape
                print(f"    {key} shape: {shape}")
                if len(shape) == 3:
                    print(f"    Format: (N_videos, T, D) = {shape}")
                elif len(shape) == 4:
                    print(f"    Format: (N_videos, T, C, D) = {shape}")
    else:
        print(f"  ✗ appearance_feat.h5: NOT FOUND")
    
    # Check motion features
    mot_path = osp.join(video_feature_path, 'motion_feat.h5')
    if osp.exists(mot_path):
        with h5py.File(mot_path, 'r') as f:
            keys = list(f.keys())
            print(f"  ✓ motion_feat.h5")
            print(f"    Keys: {keys}")
            for key in keys:
                shape = f[key].shape
                print(f"    {key} shape: {shape}")
    else:
        print(f"  ✗ motion_feat.h5: NOT FOUND")
    
    return idx2vid


def check_text_annotations(text_annotation_path, sample_vids=None):
    """Kiểm tra text annotations"""
    print("\n" + "="*60)
    print("3. CHECKING TEXT ANNOTATIONS")
    print("="*60)
    
    # Check if QA subfolder exists
    qa_path = osp.join(text_annotation_path, 'QA')
    if osp.exists(qa_path):
        print(f"  ✓ QA subfolder found")
        base_path = qa_path
    else:
        print(f"  ✗ QA subfolder NOT FOUND, checking root folder")
        base_path = text_annotation_path
    
    # Count video folders
    if osp.exists(base_path):
        video_folders = [d for d in os.listdir(base_path) if osp.isdir(osp.join(base_path, d))]
        print(f"  Found {len(video_folders)} video annotation folders")
        
        if len(video_folders) > 0:
            # Check structure of first video
            sample_vid = video_folders[0]
            vid_path = osp.join(base_path, sample_vid)
            text_file = osp.join(vid_path, 'text.json')
            answer_file = osp.join(vid_path, 'answer.json')
            
            print(f"\n  Sample video: {sample_vid}")
            
            if osp.exists(text_file):
                print(f"  ✓ text.json found")
                with open(text_file, 'r') as f:
                    text = json.load(f)
                print(f"    Keys: {list(text.keys())}")
                
                # Check question types
                expected_qtypes = ['descriptive', 'explanatory', 'predictive', 'counterfactual']
                for qtype in expected_qtypes:
                    if qtype in text:
                        q = text[qtype].get('question', 'N/A')
                        ans = text[qtype].get('answer', [])
                        print(f"    {qtype}: {len(ans)} choices")
                        if q != 'N/A':
                            print(f"      Q: {q[:50]}...")
            else:
                print(f"  ✗ text.json NOT FOUND")
            
            if osp.exists(answer_file):
                print(f"  ✓ answer.json found")
                with open(answer_file, 'r') as f:
                    answer = json.load(f)
                print(f"    Keys: {list(answer.keys())}")
                print(f"    Sample: {answer}")
            else:
                print(f"  ✗ answer.json NOT FOUND")
    
    return base_path


def check_data_alignment(split_vids, idx2vid, text_base_path):
    """Kiểm tra sự đồng bộ giữa các nguồn dữ liệu"""
    print("\n" + "="*60)
    print("4. CHECKING DATA ALIGNMENT")
    print("="*60)
    
    if idx2vid is None:
        print("  ✗ Cannot check alignment - idx2vid not loaded")
        return
    
    idx2vid_set = set(idx2vid)
    
    for split_name, vids in split_vids.items():
        vids_set = set(vids)
        
        # Check videos in split but not in visual features
        missing_visual = vids_set - idx2vid_set
        if missing_visual:
            print(f"  ⚠️  {split_name}: {len(missing_visual)} videos missing in visual features")
            print(f"      Examples: {list(missing_visual)[:3]}")
        else:
            print(f"  ✓ {split_name}: All {len(vids)} videos have visual features")
        
        # Check videos missing text annotations
        missing_text = []
        for vid in list(vids)[:100]:  # Check first 100
            text_file = osp.join(text_base_path, vid, 'text.json')
            if not osp.exists(text_file):
                missing_text.append(vid)
        
        if missing_text:
            print(f"  ⚠️  {split_name}: {len(missing_text)}/100 checked videos missing text annotations")
        else:
            print(f"  ✓ {split_name}: First 100 videos have text annotations")


def check_sample_loading(sample_list_path, video_feature_path, text_annotation_path):
    """Thử load một sample để verify DataLoader sẽ hoạt động"""
    print("\n" + "="*60)
    print("5. TESTING SAMPLE LOADING")
    print("="*60)
    
    try:
        # Load one video
        train_path = osp.join(sample_list_path, 'train.pkl')
        if not osp.exists(train_path):
            train_path = osp.join(sample_list_path, 'valid.pkl')
        
        with open(train_path, 'rb') as f:
            vids = pickle.load(f)
        
        if len(vids) == 0:
            print("  ✗ No videos in split file")
            return
        
        vid = vids[0]
        print(f"  Testing with video: {vid}")
        
        # Load idx2vid
        with open(osp.join(video_feature_path, 'idx2vid.pkl'), 'rb') as f:
            idx2vid = pickle.load(f)
        
        vid_idx = idx2vid.index(vid) if vid in idx2vid else None
        if vid_idx is None:
            print(f"  ✗ Video {vid} not found in idx2vid")
            return
        
        print(f"  Video index: {vid_idx}")
        
        # Load features
        with h5py.File(osp.join(video_feature_path, 'appearance_feat.h5'), 'r') as f:
            app_feat = f['resnet_features'][vid_idx][...]
        print(f"  Appearance shape: {app_feat.shape}")
        
        with h5py.File(osp.join(video_feature_path, 'motion_feat.h5'), 'r') as f:
            mot_feat = f['resnet_features'][vid_idx][...]
        print(f"  Motion shape: {mot_feat.shape}")
        
        # Handle dimension mismatch
        if app_feat.ndim == 3:
            app_feat = app_feat.mean(axis=1)
            print(f"  Reduced appearance to: {app_feat.shape}")
        
        # Concatenate
        frame_feat = np.concatenate([app_feat, mot_feat], axis=-1)
        print(f"  Frame feat shape: {frame_feat.shape}")
        
        # Load text
        text_file = osp.join(text_annotation_path, 'QA', vid, 'text.json')
        if not osp.exists(text_file):
            text_file = osp.join(text_annotation_path, vid, 'text.json')
        
        with open(text_file, 'r') as f:
            text = json.load(f)
        
        print(f"  Question (descriptive): {text['descriptive']['question']}")
        print(f"  Answers: {text['descriptive']['answer']}")
        
        print("\n  ✓ Sample loading successful!")
        
    except Exception as e:
        print(f"  ✗ Error during sample loading: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Check CausalVidQA data format")
    parser.add_argument('--sample_list_path', type=str, required=True,
                        help='Path to split pkl files (dataset-split-1)')
    parser.add_argument('--video_feature_path', type=str, required=True,
                        help='Path to visual features')
    parser.add_argument('--text_annotation_path', type=str, required=True,
                        help='Path to text annotations')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" CausalVidQA DATA CHECKER")
    print("="*60)
    print(f"Sample list path: {args.sample_list_path}")
    print(f"Video feature path: {args.video_feature_path}")
    print(f"Text annotation path: {args.text_annotation_path}")
    
    # Run checks
    split_vids = check_split_files(args.sample_list_path)
    idx2vid = check_visual_features(args.video_feature_path)
    text_base = check_text_annotations(args.text_annotation_path)
    
    if split_vids and idx2vid:
        check_data_alignment(split_vids, idx2vid, text_base)
    
    check_sample_loading(
        args.sample_list_path,
        args.video_feature_path, 
        args.text_annotation_path
    )
    
    print("\n" + "="*60)
    print(" CHECK COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
