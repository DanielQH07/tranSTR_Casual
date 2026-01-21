# Debug SoM Data Files
# Chạy cell này để xem nội dung thực tế của file .npz và .json

import os
import numpy as np
import json

# ============================================
# CẤU HÌNH PATH - THAY ĐỔI NẾU CẦN
# ============================================
SOM_PATH = '/kaggle/input/obj-mask-causal'  # Path tới thư mục obj_mask_causal

# ============================================
# 1. KIỂM TRA CẤU TRÚC THƯ MỤC
# ============================================
print("=" * 60)
print("1. KIỂM TRA CẤU TRÚC THƯ MỤC")
print("=" * 60)

if os.path.exists(SOM_PATH):
    print(f"✅ SOM_PATH tồn tại: {SOM_PATH}")
    print(f"\nCác thư mục con:")
    for item in os.listdir(SOM_PATH):
        item_path = os.path.join(SOM_PATH, item)
        if os.path.isdir(item_path):
            num_files = len(os.listdir(item_path))
            print(f"   📁 {item}/ ({num_files} files)")
            # Hiển thị 3 file đầu tiên
            files = os.listdir(item_path)[:3]
            for f in files:
                print(f"      - {f}")
else:
    print(f"❌ SOM_PATH không tồn tại: {SOM_PATH}")

# ============================================
# 2. KIỂM TRA FILE NPZ (id_masks)
# ============================================
print("\n" + "=" * 60)
print("2. KIỂM TRA FILE NPZ (id_masks)")
print("=" * 60)

id_masks_dir = os.path.join(SOM_PATH, 'id_masks')
if os.path.exists(id_masks_dir):
    npz_files = [f for f in os.listdir(id_masks_dir) if f.endswith('.npz')]
    print(f"Tổng số file .npz: {len(npz_files)}")
    
    # Lấy 1 file mẫu để kiểm tra
    if npz_files:
        sample_npz = npz_files[0]
        sample_path = os.path.join(id_masks_dir, sample_npz)
        print(f"\n📄 Kiểm tra file mẫu: {sample_npz}")
        
        data = np.load(sample_path)
        print(f"   Keys trong file: {list(data.keys())}")
        
        for key in list(data.keys())[:3]:  # Chỉ hiển thị 3 key đầu
            arr = data[key]
            print(f"   - {key}: shape={arr.shape}, dtype={arr.dtype}, " + 
                  f"min={arr.min()}, max={arr.max()}, unique={np.unique(arr).tolist()[:10]}")
else:
    print(f"❌ id_masks không tồn tại")

# ============================================
# 3. KIỂM TRA FILE JSON (metadata_json)
# ============================================
print("\n" + "=" * 60)
print("3. KIỂM TRA FILE JSON (metadata_json)")
print("=" * 60)

meta_dir = os.path.join(SOM_PATH, 'metadata_json')
if os.path.exists(meta_dir):
    json_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    print(f"Tổng số file .json: {len(json_files)}")
    
    if json_files:
        sample_json = json_files[0]
        sample_path = os.path.join(meta_dir, sample_json)
        print(f"\n📄 Kiểm tra file mẫu: {sample_json}")
        
        with open(sample_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"   Keys trong file: {list(metadata.keys())}")
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"   - {key}: {len(value)} items")
                # Hiển thị vài item đầu
                for k, v in list(value.items())[:3]:
                    print(f"      '{k}': {v}")
            else:
                print(f"   - {key}: {value}")
else:
    print(f"❌ metadata_json không tồn tại")

# ============================================
# 4. KIỂM TRA MATCH GIỮA VIDEO ID VÀ SoM FILES
# ============================================
print("\n" + "=" * 60)
print("4. KIỂM TRA MATCH VIDEO ID")
print("=" * 60)

# Lấy video_id từ train dataset (nếu có)
try:
    sample_vid = train_ds.sample_list.iloc[0]['video_id']
    print(f"Sample video_id từ dataset: {sample_vid}")
    
    # Kiểm tra file tương ứng
    npz_path = os.path.join(id_masks_dir, f"{sample_vid}.npz")
    json_path = os.path.join(meta_dir, f"{sample_vid}.json")
    
    print(f"\nKiểm tra file:")
    print(f"   NPZ: {npz_path}")
    print(f"   Exists: {os.path.exists(npz_path)}")
    
    print(f"   JSON: {json_path}")
    print(f"   Exists: {os.path.exists(json_path)}")
    
    # Nếu file tồn tại, hiển thị nội dung
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"\n   NPZ Keys: {list(data.keys())}")
        for key in list(data.keys())[:2]:
            arr = data[key]
            print(f"      {key}: shape={arr.shape}, unique_values={np.unique(arr).tolist()[:10]}")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
        print(f"\n   JSON Content:")
        print(json.dumps(meta, indent=4)[:500])  # Hiển thị 500 ký tự đầu
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
