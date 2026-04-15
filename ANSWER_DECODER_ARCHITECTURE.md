# Answer Decoder Architecture (TranSTR - Updated)

## 1) Muc tieu
Kien truc moi tach ro ba thanh phan trong giai doan giai ma dap an:
- Answer scoring: diem chinh cho 5 lua chon.
- Evidence scoring: do tin cay cho tung lua chon (verifier).
- Knowledge scoring: do ho tro tu tri thuc co dieu kien theo question family.

Thiet ke nay giu tuong thich nguoc voi pipeline cu (van co logits/verifier_logits), dong thoi cho phep mo rong re-rank theo knowledge.

## 2) Thanh phan chinh
Trong `VideoQAmodel`:
- `answer_head`: Linear(d_model -> 1), tao `answer_score`.
- `evidence_head`: Linear(d_model -> 1), tao `evidence_score`.
- `q_family_embed`: Embedding(num_question_families, d_model).
- `knowledge_head`: MLP tren dac trung hop nhat (cand, memory, knowledge, interactions, family embedding) de tao `knowledge_score`.
- `k_proj`: Chuan hoa knowledge feature ve `d_model` khi input knowledge co so chieu khac.

## 3) Input/Output theo API

### 3.1 `forward(...)`
Chu ky:
- `frame_feat`: Tensor [B, T, Df]
- `obj_feat`: Tensor [B, T, O, Do]
- `qns_word`: list cau hoi (hoac cau hoi da ghep theo dataset)
- `ans_word`: list 5 dap an ung vien moi mau
- `return_aux`: bool
- `q_family_id`: optional Tensor [B]
- `knowledge_feat`: optional Tensor [B, Dk] hoac [B, N, Dk]

Output:
- Neu `return_aux=False`: tra ve `logits` [B, N] (N=5).
- Neu `return_aux=True`: tra ve dict:
  - `cand_feat`: [B, N, d_model]
  - `answer_score`: [B, N]
  - `evidence_score`: [B, N]
  - `mem`: [B, L, d_model]
  - `mem_pool`: [B, d_model]
  - `logits`: [B, N] (alias cua answer_score de tuong thich nguoc)
  - `verifier_logits`: [B, N] (alias cua evidence_score)
  - Neu co `q_family_id`:
    - `knowledge_score`: [B, N]
    - `fused_score`: [B, N] = `answer_score + lambda_knowledge * knowledge_score`

### 3.2 `forward_cached(...)`
Chu ky:
- `frame_feat`: [B, T, Df]
- `obj_feat`: [B, T, O, Do]
- `text_feat`: [B, N, 768] (cached text features)
- `return_aux`, `q_family_id`, `knowledge_feat`: giong `forward(...)`

Output:
- Giong `forward(...)`.

### 3.3 `forward_with_knowledge(...)`
Chu ky:
- `frame_feat`, `obj_feat`, `qns_word`, `ans_word`, `q_family_id`, `knowledge_feat=None`

Output:
- Luon tra ve dict chi tiet (tuong duong `return_aux=True`) va dam bao co `fused_score`.

## 4) Luong tinh diem
1. Video-text fusion tao memory `mem`.
2. Answer decoder tao candidate feature `cand_feat` cho N lua chon.
3. `answer_head(cand_feat)` -> `answer_score`.
4. `evidence_head(cand_feat)` -> `evidence_score`.
5. Neu co `q_family_id`:
   - Chuan hoa `knowledge_feat` thanh [B, N, d_model].
   - Mean pool memory -> `mem_pool` [B, d_model].
   - Ghep dac trung: `[cand, mem_pool, k_feat, cand*mem_pool, cand*k_feat, q_family_embed]`.
   - `knowledge_head(...)` -> `knowledge_score`.
   - Hop nhat: `fused_score = answer_score + lambda_knowledge * knowledge_score`.

## 5) Mapping question family
Trong DataLoader da co map:
- descriptive -> 0
- explanatory -> 1
- predictive -> 2
- counterfactual -> 3
- predictive_reason -> 4
- counterfactual_reason -> 5

Khi bat `return_family_id=True`, dataset se tra them `q_family_id` o cuoi batch.

## 6) Backward compatibility
- Van giu `classifier` alias -> `answer_head`.
- Van giu `verifier` alias -> `evidence_head`.
- Van giu key output `logits` va `verifier_logits` nhu truoc.
- Neu khong truyen `q_family_id`, model se chay nhu che do cu (khong tinh knowledge_score).

## 7) Ghi chu train/eval
- Train hien tai co the dung loss:
  - `CE(logits, target)`
  - `BCE(verifier_logits, one_hot_target)`
  - `BCE(knowledge_score, one_hot_target)` (neu co knowledge_score)
- Eval nen uu tien `fused_score` neu ton tai, neu khong fallback ve `logits`.
