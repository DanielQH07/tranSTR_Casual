"""
Script để chạy train và test trên tập nhỏ CausalVidQA
Sử dụng kagglehub để download data
"""

import kagglehub
import os
import sys

# Download data từ Kaggle
print("Downloading datasets from Kaggle...")
text_feature_path = kagglehub.dataset_download('lusnaw/text-feature')
visual_feature_path = kagglehub.dataset_download('lusnaw/visual-feature')
split_path = kagglehub.dataset_download('lusnaw/dataset-split-1')
text_annotation_path = kagglehub.dataset_download('lusnaw/text-annotation')

print(f"\nData paths:")
print(f"  Text features: {text_feature_path}")
print(f"  Visual features: {visual_feature_path}")
print(f"  Split files: {split_path}")
print(f"  Text annotations: {text_annotation_path}")

# ============================================================
# TRAIN COMMAND (small test - 2 epochs, batch size 4, 10 videos)
# ============================================================
train_cmd = f'''
python train.py \\
    -v small_test \\
    -bs 4 \\
    -lr 1e-4 \\
    -epoch 2 \\
    -gpu 0 \\
    --sample_list_path "{split_path}" \\
    --video_feature_path "{visual_feature_path}" \\
    --text_annotation_path "{text_annotation_path}" \\
    --qtype -1 \\
    --max_samples 10 \\
    -fk 4 \\
    -ok 5 \\
    -objs 10 \\
    -el 1 \\
    -dl 1 \\
    -t microsoft/deberta-base
'''

# ============================================================
# TEST COMMAND
# ============================================================
test_cmd = f'''
python test.py \\
    -v small_test \\
    -bs 4 \\
    -gpu 0 \\
    --sample_list_path "{split_path}" \\
    --video_feature_path "{visual_feature_path}" \\
    --text_annotation_path "{text_annotation_path}" \\
    --qtype -1 \\
    --max_samples 10 \\
    -fk 4 \\
    -ok 5 \\
    -objs 10 \\
    -el 1 \\
    -dl 1 \\
    -t microsoft/deberta-base \\
    --model_path "./models/best_model-small_test_XXX.ckpt"
'''

print("\n" + "="*60)
print("TRAIN COMMAND (copy and run in terminal):")
print("="*60)
print(train_cmd.replace('\\', '').replace('\n', ' '))

print("\n" + "="*60)
print("TEST COMMAND (update model_path first):")
print("="*60)
print(test_cmd.replace('\\', '').replace('\n', ' '))

# ============================================================
# Direct execution option
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Actually run the training')
    parser.add_argument('--max_samples', type=int, default=10, help='Number of videos to use')
    args = parser.parse_args()
    
    if args.run:
        # Update command with custom max_samples
        actual_cmd = train_cmd.replace('--max_samples 10', f'--max_samples {args.max_samples}')
        print("\n" + "="*60)
        print(f"Starting training with {args.max_samples} videos...")
        print("="*60)
        os.system(actual_cmd.replace('\\', '').replace('\n', ' '))
