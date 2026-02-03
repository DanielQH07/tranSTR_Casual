# ==============================================================================
# UPDATED VISUALIZATION CODE FOR inference.ipynb
# Copy these cells to replace CELL 11, 14, 15, and 16 in the notebook
# ==============================================================================

# ==============================================================================
# CELL 11: Visualization Functions (REPLACE THIS CELL)
# ==============================================================================
"""
# Copy the following into CELL 11:

# ==============================================================================
# CELL 11: Visualization Functions
# ==============================================================================
print('=== CELL 11: Visualization ===')

# Use fonts that support needed characters
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def visualize_sample(sample_data, result_som, result_no_som, video_frames=None, som_masks=None):
    '''
    Create comprehensive visualization for a sample.
    Shows frames, Q&A with clear correct/wrong labels, masks overlay.
    '''
    qns_key = sample_data['qns_key']
    question = sample_data['question']
    answers = sample_data['answers']
    correct_ans = sample_data['correct_ans']
    entity_names = sample_data.get('entity_names', {})
    
    som_correct = result_som['pred'] == correct_ans
    no_som_correct = result_no_som['pred'] == correct_ans
    
    # Create figure with better layout - 5 rows
    fig = plt.figure(figsize=(24, 22))
    
    # Title with clear status (no emoji)
    som_status = 'CORRECT' if som_correct else 'WRONG'
    no_som_status = 'CORRECT' if no_som_correct else 'WRONG'
    title_color = 'green' if som_correct else 'red'
    fig.suptitle(f"Sample: {qns_key}\\nSoM: [{som_status}] | No-SoM: [{no_som_status}]", 
                 fontsize=16, fontweight='bold', color=title_color)
    
    # ============================================
    # Row 1-2: 16 Sampled Frames (2 rows x 8 cols)
    # ============================================
    if video_frames is not None:
        for i in range(min(16, len(video_frames))):
            ax = fig.add_subplot(5, 8, i + 1)
            ax.imshow(video_frames[i])
            
            # Highlight selected frames with green border
            is_selected = i in result_som['selected_frames']
            if is_selected:
                for spine in ax.spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(4)
                ax.set_title(f'F{i} [SEL]', fontsize=9, color='lime', fontweight='bold')
            else:
                ax.set_title(f'F{i}', fontsize=9, color='gray')
            
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax = fig.add_subplot(5, 1, 1)
        ax.text(0.5, 0.5, 'Video frames not available', 
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
    
    # ============================================
    # Row 3: Selected Frames with Masks Overlay
    # ============================================
    selected_frames = result_som['selected_frames']
    num_selected = len(selected_frames)
    
    for idx, frame_idx in enumerate(selected_frames[:8]):  # Max 8 frames
        ax = fig.add_subplot(5, 8, 17 + idx)
        
        if video_frames is not None and frame_idx < len(video_frames):
            ax.imshow(video_frames[frame_idx])
            
            # Overlay mask if available
            if som_masks is not None and frame_idx in som_masks:
                mask = som_masks[frame_idx]
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                
                # Resize mask to match frame if needed
                frame_h, frame_w = video_frames[frame_idx].shape[:2]
                if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize((frame_w, frame_h), PILImage.NEAREST)
                    mask = np.array(mask_img)
                
                # Create colored overlay for each entity
                overlay = np.zeros((frame_h, frame_w, 4), dtype=np.float32)
                cmap = plt.cm.get_cmap('tab10')
                unique_ids = np.unique(mask)
                for eid in unique_ids:
                    if eid > 0:  # Skip background
                        color = cmap(int(eid) % 10)
                        entity_mask = mask == eid
                        overlay[entity_mask] = [color[0], color[1], color[2], 0.5]
                
                ax.imshow(overlay)
                ax.set_title(f'F{frame_idx} + Mask', fontsize=9, color='orange', fontweight='bold')
            else:
                ax.set_title(f'F{frame_idx} (no mask)', fontsize=9, color='gray')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            ax.set_title(f'F{frame_idx}', fontsize=9)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
    
    # ============================================
    # Row 4: Q&A Display with Clear Labels
    # ============================================
    ax_qa = fig.add_subplot(5, 2, 7)
    ax_qa.axis('off')
    
    # Build Q&A text with clear markers (NO emoji)
    qa_lines = []
    qa_lines.append(f"QUESTION: {question}")
    qa_lines.append("")
    qa_lines.append("ANSWERS:")
    
    for i, ans in enumerate(answers):
        markers = []
        
        if i == correct_ans:
            markers.append('<<CORRECT>>')
        
        if i == result_som['pred']:
            if i == correct_ans:
                markers.append('[SoM-OK]')
            else:
                markers.append('[SoM-WRONG]')
        
        if i == result_no_som['pred']:
            if i == correct_ans:
                markers.append('[NoSoM-OK]')
            else:
                markers.append('[NoSoM-WRONG]')
        
        marker_str = ' '.join(markers)
        ans_display = ans[:60] + '...' if len(ans) > 60 else ans
        line = f"  [{i}] {ans_display}"
        if marker_str:
            line += f"  >>> {marker_str}"
        qa_lines.append(line)
    
    qa_text = '\\n'.join(qa_lines)
    
    ax_qa.text(0.02, 0.95, qa_text, transform=ax_qa.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='black'))
    
    # ============================================
    # Row 4: Prediction Probabilities Bar Chart
    # ============================================
    ax_comp = fig.add_subplot(5, 2, 8)
    
    x = np.arange(5)
    width = 0.35
    
    bars1 = ax_comp.bar(x - width/2, result_som['probs'], width, label='With SoM', 
                        color='forestgreen', alpha=0.7, edgecolor='black')
    bars2 = ax_comp.bar(x + width/2, result_no_som['probs'], width, label='Without SoM', 
                        color='coral', alpha=0.7, edgecolor='black')
    
    # Highlight correct answer bar with gold border
    bars1[correct_ans].set_edgecolor('gold')
    bars1[correct_ans].set_linewidth(3)
    bars2[correct_ans].set_edgecolor('gold')
    bars2[correct_ans].set_linewidth(3)
    
    ax_comp.set_ylabel('Probability')
    ax_comp.set_xlabel('Answer Option')
    ax_comp.set_title('Prediction Probabilities (Gold border = Correct Answer)')
    ax_comp.set_xticks(x)
    labels = [f'A{i}' + (' [GT]' if i == correct_ans else '') for i in range(5)]
    ax_comp.set_xticklabels(labels)
    ax_comp.legend(loc='upper right')
    ax_comp.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax_comp.set_ylim(0, 1.0)
    
    # ============================================
    # Row 5: Frame Attention Weights
    # ============================================
    ax_att = fig.add_subplot(5, 2, 9)
    
    frame_w = result_som['frame_weights']
    colors = ['lime' if i in result_som['selected_frames'] else 'lightgray' for i in range(len(frame_w))]
    ax_att.bar(range(len(frame_w)), frame_w, color=colors, edgecolor='black')
    ax_att.set_xlabel('Frame Index')
    ax_att.set_ylabel('Attention Weight')
    ax_att.set_title(f'Frame Attention (Green = Selected Top-{num_selected})')
    ax_att.set_xticks(range(len(frame_w)))
    
    # ============================================
    # Row 5: Summary Box
    # ============================================
    ax_sum = fig.add_subplot(5, 2, 10)
    ax_sum.axis('off')
    
    som_pred_conf = result_som['probs'][result_som['pred']] * 100
    no_som_pred_conf = result_no_som['probs'][result_no_som['pred']] * 100
    
    summary_lines = [
        "=" * 45,
        "RESULTS SUMMARY",
        "=" * 45,
        "",
        "With Token Mark (SoM):",
        f"  Prediction: {result_som['pred']}  Status: {'CORRECT' if som_correct else 'WRONG'}",
        f"  Confidence: {som_pred_conf:.1f}%",
        f"  Selected Frames: {list(selected_frames)}",
        "",
        "Without Token Mark:",
        f"  Prediction: {result_no_som['pred']}  Status: {'CORRECT' if no_som_correct else 'WRONG'}",
        f"  Confidence: {no_som_pred_conf:.1f}%",
        "",
        "=" * 45,
        f"Ground Truth Answer: {correct_ans}",
        "=" * 45,
    ]
    
    if entity_names:
        summary_lines.append("")
        summary_lines.append("Detected Entities:")
        for eid, ename in entity_names.items():
            summary_lines.append(f"  [{eid}] {ename}")
    
    summary = '\\n'.join(summary_lines)
    
    # Color based on SoM result
    bg_color = 'lightgreen' if som_correct else 'lightcoral'
    ax_sum.text(0.02, 0.95, summary, transform=ax_sum.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.7, edgecolor='black'))
    
    plt.subplots_adjust(top=0.93, bottom=0.05, hspace=0.4, wspace=0.2)
    return fig

print('Visualization function defined')
"""

# ==============================================================================
# CELL 14: Visualize All Samples (REPLACE THIS CELL)
# ==============================================================================
"""
# Copy the following into CELL 14:

# ==============================================================================
# CELL 14: Visualize All Samples
# ==============================================================================
print('=== CELL 14: Visualizations ===')

for i, res in enumerate(results_list):
    print(f'\\nVisualizing sample {i+1}/{len(results_list)}: {res["sample_data"]["qns_key"]}')
    
    fig = visualize_sample(
        res['sample_data'],
        res['result_som'],
        res['result_no_som'],
        res['video_frames'],
        res['som_masks']
    )
    
    # Save figure
    fig.savefig(f'inference_sample_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

print('\\nAll visualizations complete!')
"""

# ==============================================================================
# CELL 15: Summary Statistics (REPLACE THIS CELL)
# ==============================================================================
"""
# Copy the following into CELL 15:

# ==============================================================================
# CELL 15: Summary Statistics
# ==============================================================================
print('=== CELL 15: Summary ===')

som_correct = sum(1 for r in results_list if r['result_som']['pred'] == r['sample_data']['correct_ans'])
no_som_correct = sum(1 for r in results_list if r['result_no_som']['pred'] == r['sample_data']['correct_ans'])
total = len(results_list)

print('\\n' + '='*60)
print('INFERENCE SUMMARY')
print('='*60)
print(f'Total samples: {total}')
print(f'\\nWith Token Mark (SoM):')
print(f'  Correct: {som_correct}/{total} ({som_correct/total*100:.1f}%)')
print(f'\\nWithout Token Mark:')
print(f'  Correct: {no_som_correct}/{total} ({no_som_correct/total*100:.1f}%)')
print('='*60)

# Detailed breakdown (no emoji, use text markers)
print('\\nPer-sample breakdown:')
print('-'*60)
for i, res in enumerate(results_list):
    som_ok = 'OK' if res['result_som']['pred'] == res['sample_data']['correct_ans'] else 'WRONG'
    no_som_ok = 'OK' if res['result_no_som']['pred'] == res['sample_data']['correct_ans'] else 'WRONG'
    qkey = res['sample_data']['qns_key'][:35]
    print(f"{i+1}. {qkey:<35} SoM:[{som_ok:5s}] No-SoM:[{no_som_ok:5s}]")

# Log to W&B
wandb.log({
    'inference/som_accuracy': som_correct/total*100,
    'inference/no_som_accuracy': no_som_correct/total*100,
    'inference/samples': total
})

wandb.finish()
print('\\nDone!')
"""

# ==============================================================================
# CELL 16: Display 16 Frames with Entity Masks (REPLACE THIS CELL)
# ==============================================================================
"""
# Copy the following into CELL 16:

# ==============================================================================
# CELL 16: Entity Mask Visualization
# ==============================================================================
print('=== CELL 16: Entity Mask Visualization ===')

# Select a sample with SoM data
sample_with_som = None
for res in results_list:
    if res['som_masks'] and res['video_frames'] is not None:
        sample_with_som = res
        break

if sample_with_som:
    print(f"Visualizing entity masks for: {sample_with_som['sample_data']['qns_key']}")
    
    frames = sample_with_som['video_frames']
    masks = sample_with_som['som_masks']
    entities = sample_with_som['sample_data'].get('entity_names', {})
    selected_frames = sample_with_som['result_som']['selected_frames']
    
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.flatten()
    
    # Define colors for entities
    cmap = plt.cm.get_cmap('tab10')
    
    for i in range(16):
        ax = axes[i]
        
        if i < len(frames):
            ax.imshow(frames[i])
            
            # Overlay mask if available
            if i in masks:
                mask = masks[i]
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                
                # Resize mask to match frame
                frame_h, frame_w = frames[i].shape[:2]
                if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize((frame_w, frame_h), PILImage.NEAREST)
                    mask = np.array(mask_img)
                
                # Create colored overlay
                overlay = np.zeros((frame_h, frame_w, 4), dtype=np.float32)
                for entity_id in np.unique(mask):
                    if entity_id > 0:  # Skip background
                        color = cmap(int(entity_id) % 10)
                        entity_mask = mask == entity_id
                        overlay[entity_mask] = [color[0], color[1], color[2], 0.4]
                
                ax.imshow(overlay)
        
        # Highlight selected frames
        if i in selected_frames:
            for spine in ax.spines.values():
                spine.set_edgecolor('lime')
                spine.set_linewidth(4)
            ax.set_title(f'F{i} [SEL]', fontsize=10, color='lime', fontweight='bold')
        else:
            ax.set_title(f'F{i}', fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Legend for entities
    if entities:
        legend_parts = [f'[{k}]: {v}' for k, v in entities.items()]
        legend_text = 'Detected Entities: ' + ', '.join(legend_parts)
        fig.text(0.5, 0.02, legend_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f"16 Frames with Entity Masks - {sample_with_som['sample_data']['qns_key']}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig('frames_with_masks.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No sample with both video frames and SoM masks found.')

print('\\nEntity mask visualization complete!')
"""

print("=" * 70)
print("INSTRUCTIONS: Copy the code blocks above into your inference.ipynb")
print("Replace CELL 11, 14, 15, and 16 with the corresponding code blocks")
print("=" * 70)
