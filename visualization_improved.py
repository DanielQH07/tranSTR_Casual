"""
IMPROVED VISUALIZATION FUNCTION for inference.ipynb

Copy this code to replace CELL 11 in inference.ipynb.
This version separates visualizations into multiple clear figures
(similar to inference_wandb.ipynb style) for better readability.

Changes from original:
1. Separates into 5 distinct figures instead of cramming into 1
2. Shows 16 frames in clear 2x8 grid with scores
3. Temporal bar chart with threshold and highlighting
4. Spatial view with bounding boxes on selected frames
5. Q&A results with better formatting
6. Probability comparison chart with percentage labels
"""

# ==============================================================================
# CELL 11: Visualization Functions (IMPROVED)
# ==============================================================================
CELL_11_CODE = '''
# ==============================================================================
# CELL 11: Visualization Functions
# ==============================================================================
print('=== CELL 11: Visualization ===')

def visualize_sample(sample_data, result_som, result_no_som, video_frames=None, som_masks=None):
    """
    Create comprehensive visualization for a sample.
    Split into multiple clear figures for better readability.
    """
    qns_key = sample_data['qns_key']
    question = sample_data['question']
    answers = sample_data['answers']
    correct_ans = sample_data['correct_ans']
    
    print(f'\\n{"="*60}')
    print(f'Sample: {qns_key}')
    print('='*60)
    
    all_figs = []
    
    # ============================================
    # FIGURE 1: 16 Sampled Frames (2x8 grid)
    # ============================================
    if video_frames is not None and len(video_frames) > 0:
        fig1, axes = plt.subplots(2, 8, figsize=(20, 6))
        fig1.suptitle(f'16 Sampled Frames - {qns_key}', fontsize=14, fontweight='bold')
        
        for i, ax in enumerate(axes.flatten()):
            if i < len(video_frames):
                ax.imshow(video_frames[i])
                # Get frame score
                score = result_som['frame_weights'][i] if i < len(result_som['frame_weights']) else 0
                # Check if frame is selected
                is_selected = i in result_som['selected_frames']
                # Color based on selection
                title_color = 'green' if is_selected else 'gray'
                ax.set_title(f'F{i} ({score:.2f})', color=title_color, fontsize=10, 
                             fontweight='bold' if is_selected else 'normal')
                # Add border for selected frames
                if is_selected:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('lime')
                        spine.set_linewidth(3)
            ax.axis('off')
        
        plt.tight_layout()
        all_figs.append(fig1)
        plt.show()
    else:
        print('⚠️ Video frames not available')
    
    # ============================================
    # FIGURE 2: Temporal Rationalization (Bar Chart)
    # ============================================
    fig2, ax = plt.subplots(figsize=(14, 4))
    
    scores = result_som['frame_weights']
    num_frames = len(scores)
    selected_frames = result_som['selected_frames']
    
    # Color bars: selected = green, others = lightgray
    colors = ['limegreen' if i in selected_frames else 'lightgray' for i in range(num_frames)]
    bars = ax.bar(range(num_frames), scores, color=colors, edgecolor='black', linewidth=1)
    
    # Highlight selected frames with thicker border
    for idx in selected_frames:
        if idx < len(bars):
            bars[idx].set_edgecolor('darkgreen')
            bars[idx].set_linewidth(3)
    
    # Threshold line
    threshold = np.median(scores)
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Median ({threshold:.2f})')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Attention Score', fontsize=12)
    ax.set_title(f'Temporal Rationalization - Top {len(selected_frames)} Frames Selected (Green)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_frames))
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    all_figs.append(fig2)
    plt.show()
    
    # ============================================
    # FIGURE 3: Spatial Rationalization (Selected Frames + Bboxes)
    # ============================================
    if video_frames is not None and len(video_frames) > 0:
        selected_frame_indices = sorted([i for i in selected_frames if i < len(video_frames)])
        if len(selected_frame_indices) > 0:
            n_selected = len(selected_frame_indices)
            fig3, axes = plt.subplots(1, n_selected, figsize=(4 * n_selected, 4))
            if n_selected == 1:
                axes = [axes]
            fig3.suptitle('Spatial Rationalization - Selected Frames with Object Boxes', 
                          fontsize=12, fontweight='bold')
            
            for ax_idx, frame_idx in enumerate(selected_frame_indices):
                ax = axes[ax_idx]
                ax.imshow(video_frames[frame_idx])
                
                # Draw object bounding boxes if available
                if 'selected_objects' in result_som and ax_idx < len(result_som['selected_objects']):
                    obj_indices = result_som['selected_objects'][ax_idx]
                    if 'obj_bboxes' in result_som and frame_idx < len(result_som.get('obj_bboxes', [])):
                        bboxes = result_som['obj_bboxes'][frame_idx]
                        for obj_idx in obj_indices:
                            if obj_idx < len(bboxes):
                                x1, y1, x2, y2 = bboxes[obj_idx][:4]
                                if x2 > x1 and y2 > y1:  # Valid bbox
                                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                         fill=False, edgecolor='lime', linewidth=2)
                                    ax.add_patch(rect)
                
                ax.set_title(f'Frame {frame_idx}', fontsize=11, fontweight='bold')
                ax.axis('off')
            
            plt.tight_layout()
            all_figs.append(fig3)
            plt.show()
    
    # ============================================
    # FIGURE 4: Q&A Results
    # ============================================
    fig4, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Build Q&A text
    text = f"Question:\\n{question}\\n\\n"
    text += "Answers:\\n"
    
    for i, ans in enumerate(answers):
        prefix = ''
        suffix = ''
        
        # Mark correct answer
        if i == correct_ans:
            prefix = '✓ CORRECT '
        
        # Mark predictions
        if i == result_som['pred']:
            suffix += ' <<< SoM'
        if i == result_no_som['pred']:
            suffix += ' <<< No-SoM'
        
        # Probability
        prob_som = result_som['probs'][i] * 100
        prob_no_som = result_no_som['probs'][i] * 100
        
        text += f"  {prefix}({i}) {ans} [SoM:{prob_som:.1f}% | No-SoM:{prob_no_som:.1f}%]{suffix}\\n"
    
    # Result summary
    som_correct = result_som['pred'] == correct_ans
    no_som_correct = result_no_som['pred'] == correct_ans
    
    text += f"\\n{'─'*50}\\n"
    text += f"Prediction SoM: {result_som['pred']} | GT: {correct_ans} | {'✅ CORRECT' if som_correct else '❌ WRONG'}\\n"
    text += f"Prediction No-SoM: {result_no_som['pred']} | GT: {correct_ans} | {'✅ CORRECT' if no_som_correct else '❌ WRONG'}"
    
    # Background color based on correctness
    bg_color = 'lightgreen' if som_correct else 'lightcoral'
    
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.6))
    
    ax.set_title(f'Q&A Results - {qns_key}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    all_figs.append(fig4)
    plt.show()
    
    # ============================================
    # FIGURE 5: Prediction Probability Comparison
    # ============================================
    fig5, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(5)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, result_som['probs'], width, label='With SoM', 
                   color='forestgreen', alpha=0.8, edgecolor='darkgreen')
    bars2 = ax.bar(x + width/2, result_no_som['probs'], width, label='Without SoM', 
                   color='indianred', alpha=0.8, edgecolor='darkred')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xlabel('Answer Option', fontsize=12)
    ax.set_title('Prediction Probabilities Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'A{i}' for i in range(5)])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Random (20%)')
    
    # Highlight correct answer tick
    ax.get_xticklabels()[correct_ans].set_color('blue')
    ax.get_xticklabels()[correct_ans].set_fontweight('bold')
    ax.get_xticklabels()[correct_ans].set_fontsize(12)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    all_figs.append(fig5)
    plt.show()
    
    return all_figs

print('✅ Visualization function defined')
'''

print("="*70)
print("HOW TO USE:")
print("="*70)
print("""
1. Open inference.ipynb in Jupyter/Kaggle
2. Find CELL 11 (Visualization Functions)
3. Delete ALL the existing code in that cell
4. Copy the code between the ''' marks above and paste into the cell
5. Run the cell

The new visualization will show:
- Fig 1: 16 frames in 2x8 grid with scores and highlighting
- Fig 2: Temporal bar chart showing frame selection
- Fig 3: Selected frames with object bounding boxes
- Fig 4: Q&A results with formatted text
- Fig 5: Probability comparison bar chart

Each figure is displayed separately for clarity!
""")
print("="*70)
