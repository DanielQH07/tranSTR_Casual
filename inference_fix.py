"""
FIXED INFERENCE FUNCTION for inference_wandb.ipynb

Original bug: The code assumed frame_att had shape [B, nheads, F, q_len] (4D)
Reality: MultiheadAttention.forward returns `weights.mean(1)` which averages 
         over heads, so frame_att is actually [B, F, q_len] (3D)

The error "Dimension out of range (expected [-3, 2], but got 3)" happened because
we tried to access dim 3 on a 3D tensor.

To FIX this in your notebook, REPLACE the entire CELL 5 source with the code below:
"""

# CELL 5: Inference Function (FIXED v2)
CELL_5_FIXED_CODE = '''
# CELL 5: Inference Function (FIXED v2)
print('=== CELL 5: Inference Function ===')
from utils.util import transform_bb

@torch.no_grad()
def inference_with_attention(model, video_id, qtype, device):
    # Load ViT features
    ff = torch.load(os.path.join(VIT_FEATURE_PATH, f'{video_id}.pt'), weights_only=True)
    if isinstance(ff, np.ndarray): ff = torch.from_numpy(ff)
    ff = ff.float()
    nf = ff.shape[0]
    if nf > args.topK_frame:
        indices = np.linspace(0, nf - 1, args.topK_frame).astype(int)
        ff = ff[indices]
    elif nf < args.topK_frame:
        ff = torch.cat([ff, torch.zeros(args.topK_frame - nf, ff.shape[1])], 0)
    
    # Load object features
    obj_pkl = None
    for subdir in os.listdir(OBJ_FEATURE_PATH):
        subdir_path = os.path.join(OBJ_FEATURE_PATH, subdir)
        if os.path.isdir(subdir_path):
            pkl_path = os.path.join(subdir_path, f'{video_id}.pkl')
            if os.path.exists(pkl_path): obj_pkl = pkl_path; break
    
    objs, obj_bboxes = [], []
    if obj_pkl:
        with open(obj_pkl, 'rb') as f: data = pickle.load(f)
        feats, bboxes = np.array(data.get('features')), np.array(data.get('bboxes'))
        num_frames = feats.shape[0]
        obj_indices = np.linspace(0, num_frames - 1, args.topK_frame).astype(int) if num_frames > args.topK_frame else range(num_frames)
        for i in obj_indices:
            feat, bbox = torch.from_numpy(feats[i]).float(), torch.from_numpy(bboxes[i]).float()
            if feat.shape[0] > args.objs: feat, bbox = feat[:args.objs], bbox[:args.objs]
            elif feat.shape[0] < args.objs:
                p = args.objs - feat.shape[0]
                feat = torch.cat([feat, torch.zeros(p, feat.shape[1])], 0)
                bbox = torch.cat([bbox, torch.zeros(p, bbox.shape[1])], 0)
            bb = torch.from_numpy(transform_bb(bbox.numpy(), 640, 480)).float()
            objs.append(torch.cat([feat, bb], -1))
            obj_bboxes.append(bbox.numpy())
    while len(objs) < args.topK_frame:
        objs.append(torch.zeros(args.objs, 2053))
        obj_bboxes.append(np.zeros((args.objs, 4)))
    of = torch.stack(objs)
    
    # Load QA data
    qa_data = load_qa_data(video_id, ANNOTATION_PATH)
    if qtype not in qa_data: raise ValueError(f'{qtype} not found for {video_id}')
    qa = qa_data[qtype]
    qns, choices, correct_idx = qa['question'], qa['choices'], qa['correct_idx']
    # Create answer strings for each choice
    ans_strings = [f"{qns} [SEP] {c}" for c in choices]
    
    # Prepare tensors
    ff, of = ff.unsqueeze(0).to(device), of.unsqueeze(0).to(device)
    B, F, O = of.size()[:3]
    
    # =========================================================================
    # FIX v2: Handle frame_att correctly - it's 3D [B, F, q_len], not 4D!
    # The attention module returns weights.mean(1) which averages over heads
    # =========================================================================
    frame_feat = model.frame_resize(ff)
    q_local, q_mask = model.forward_text([qns], device)
    frame_mask = torch.ones(B, F).bool().to(device)
    
    # Get attention from decoder
    # frame_att shape: [B, F (queries), q_len (keys)] - already head-averaged
    frame_local, frame_att = model.frame_decoder(
        frame_feat, q_local, 
        memory_key_padding_mask=q_mask, 
        query_pos=model.pos_encoder_1d(frame_mask, model.d_model), 
        output_attentions=True
    )
    
    # FIX: frame_att is 3D [B, F, q_len] - mean over q_len to get per-frame importance
    frame_importance = frame_att.mean(dim=-1)  # [B, F]
    selected_frame_scores = frame_importance[0].cpu().numpy()
    
    # Get top-K frame indices for visualization
    topk_frame_indices = torch.topk(frame_importance[0], k=min(args.select_frames, F)).indices.cpu().numpy()
    
    # =========================================================================
    # Compute object scores using feature norm as proxy for importance
    # =========================================================================
    obj_scores_list = []
    for fidx in range(min(args.select_frames, len(obj_bboxes))):
        # Get object features for this frame
        obj_feat_frame = of[0, fidx]  # [O, obj_feat_dim]
        # Use L2 norm as importance score (non-zero objects have higher norm)
        obj_importance = obj_feat_frame.norm(dim=-1).cpu().numpy()  # [O]
        obj_scores_list.append(obj_importance)
    selected_obj_scores = np.array(obj_scores_list) if obj_scores_list else np.zeros((args.select_frames, O))
    
    # Run full model forward for prediction
    # CRITICAL FIX 1: Pass [qns] as a list (batch of 1 question)
    # CRITICAL FIX 2: Pass [ans_strings] as list of lists (batch of 1, each with 5 answers)
    #   model.forward uses chain(*ans_word) which needs list of lists, not flat list!
    #   With flat list ["str1", "str2"], chain iterates CHARACTERS
    #   With nested list [["str1", "str2"]], chain gives ["str1", "str2"]
    ans_word_batch = [tuple(ans_strings)]  # List containing 1 tuple of 5 answers
    out = model(ff, of, [qns], ans_word_batch)
    pred = out.argmax(-1).item()
    probs = torch.softmax(out, dim=-1)[0].cpu().numpy()
    
    return {
        'prediction': pred, 
        'probs': probs, 
        'correct_idx': correct_idx, 
        'question': qns, 
        'choices': choices, 
        'frame_scores': selected_frame_scores, 
        'obj_scores': selected_obj_scores, 
        'obj_bboxes': obj_bboxes,
        'topk_frame_indices': topk_frame_indices
    }

print('Inference function defined')
'''

print("=" * 70)
print("COPY THE CODE BETWEEN THE ''' MARKS INTO CELL 5 OF YOUR NOTEBOOK")
print("=" * 70)
print("\nThe issue was:")
print("1. frame_att shape is [B, F, q_len] (3D) - already head-averaged")
print("2. Original code assumed it was [B, nheads, F, q_len] (4D)")
print("3. Tried mean(dim=(1,3)) which failed on 3D tensor")
print("\nThe FIX:")
print("- Use frame_att.mean(dim=-1) to get per-frame importance scores")
print("- This correctly averages over q_len dimension")

