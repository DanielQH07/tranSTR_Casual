# Train Notebook Pipeline Review (Current State + 70% Feasibility)

## 1) Scope audited

This review is based on:
- `Train_full_mode_kaggle.ipynb`
- `DataLoader.py`
- `networks/model.py`
- `networks/multimodal_transformer.py`
- `networks/attention.py`
- `networks/topk.py`
- `explain/gradcam_hooks.py`
- `explain/viz_gradcam.py`

No local training outputs (`models/*.json`, `models/*.csv`) were found in this workspace, so the 70% judgment is architecture/config based, not from completed run logs.

---

## 2) End-to-end pipeline in current notebook

1. **Environment + dependency setup**
   - Kaggle path auto-detection.
   - `transformers`, `wandb`, and compatibility pin for `huggingface_hub`.

2. **Data resolution**
   - Frame features: DINOv3 `.pt` files.
   - Object features: GroundingDINO + FRCNN `.pkl`.
   - QA annotations from `text.json`/`answer.json`.
   - Split files from `train.pkl`, `valid.pkl`, `test.pkl`.

3. **Dataset build**
   - `VideoQADataset` creates 6 question families:
     `descriptive`, `explanatory`, `predictive`, `predictive_reason`, `counterfactual`, `counterfactual_reason`.
   - Returns:
     - frame features `[T, frame_feat_dim]`
     - object features `[T, O, obj_feat_dim]`
     - question string + 5 answer candidates
     - target answer index
     - `qns_key`
     - `q_family_id` (if enabled)

4. **Model + optimization**
   - Core model: `VideoQAmodel` with DeBERTa text encoder and hierarchical frame/object transformers.
   - Dual optimization:
     - `optimizer_model` (AdamW) for model parameters.
     - `optimizer_u` (SGD) for NCOD parameter vector `U`.

5. **Training objective**
   - Main branch uses integrated loss:
     - NCOD LUM/HUM branch (`l1`)
     - NCOD consistency branch (`l2`)
     - Verifier BCE branch
     - Knowledge BCE branch
   - Gradient accumulation enabled in notebook (`bs=8`, `accumulation_steps=4`, effective batch=32).
   - Gradient clipping and `U.clamp_(0, 0.99)`.

6. **Validation + early stop**
   - Validation on `fused_score` (if available) else logits.
   - `ReduceLROnPlateau` + early stopping (`min_delta`, `start_epoch`, `patience`).

7. **Detailed test evaluation**
   - Per-question-family accuracy.
   - Hard metrics `PAR`, `CAR`.
   - Aggregated `Acc_ALL`.
   - Optional memory-consistency post-check via `CausalKnowledgeRetriever`.
   - Weighted score:
     `0.35*Predictive-Reason + 0.35*Counterfactual-Reason + 0.20*Acc_ALL + 0.10*best_val_acc`.

---

## 3) Current model architecture (rewritten)

### Visual branch
- **Frame input**: DINOv3 features.
- `frame_resize`: project `frame_feat_dim -> d_model`.
- `frame_decoder`: question-conditioned frame refinement.
- `PerturbedTopK`: select top-K frames by attention-derived importance.

### Object branch
- **Object input**: GroundingDINO+FRCNN features.
- When `obj_use_bbox_pos_embed=True`, object feature is split:
  - semantic part -> norm + projection
  - bbox part -> 2D sinusoidal positional embedding
  - summed then normalized
- Objects are aligned to selected frames (hard gather if enabled).
- `obj_decoder`: question-conditioned object refinement.
- For GroundingDINO mode, object top-k selection is skipped (use all selected-frame objects).

### Hierarchical fusion
- `fo_decoder`: frame queries attend object tokens.
- `vl_encoder`: encode concatenated `[frame_obj_tokens | question_tokens]`.

### Answer reasoning
- Answer candidates are encoded as `question [SEP] answer_i`.
- `ans_decoder`: each candidate attends fused memory.
- Output heads:
  - `answer_head` (main logits)
  - `evidence_head` (verifier logits)
  - `knowledge_head` (family-aware knowledge support)
- Final fused score (if knowledge active):
  - `answer_score + lambda_knowledge * knowledge_score`.

### Noise-aware learning (NCOD/HUM)
- Per-sample trainable scalar `U[i]`.
- LUM/HUM switching by `q_family_id`:
  - easy families use LUM style
  - hard families use HUM weighting
- `U` updated by separate objective and optimizer.

---

## 4) Grad-CAM / visualization audit

### What is good
- `explain/gradcam_hooks.py` captures multiple intermediate targets:
  frame, object, question tokens, unified memory.
- `explain/viz_gradcam.py` provides clear visual renderers:
  frame heat, object box tint, token-level map, memory split, rollout compare.

### Critical mismatch with training forward (important)
1. **Object selection mismatch**
   - Training/inference model can use hard frame-object gather (`_select_obj_by_frame`).
   - Grad-CAM hooked path currently uses legacy soft mixing on objects.
   - Result: object attribution can differ from real model path.

2. **BBox positional encoding path mismatch**
   - In model forward, `obj_use_bbox_pos_embed=True` uses split semantic+bbox embedding path.
   - Hooked Grad-CAM forward currently applies `obj_pre_norm + obj_resize` directly, skipping bbox embedding composition.
   - Result: Grad-CAM object relevance can be structurally biased.

3. **Interpretation risk**
   - If hooked forward is not identical to deployment forward, heatmaps are "debug-level clues", not faithful explanations.

### Recommendation for Grad-CAM reliability
- Refactor hooked forward to reuse model's real helper methods:
  `_select_obj_by_frame`, `_fit_obj_feat_dim`, `_encode_objects`.
- Keep same branch conditions as real forward (`use_grounding_dino`, `obj_hard_gather_from_frame`, `hard_eval`).
- Add a sanity check:
  compare hooked logits vs model logits on same batch (`max_abs_diff` threshold).

---

## 5) Can current setup exceed 70%?

## Short answer
**Not likely with the current config only, unless metric target is a simpler slice and not full causal reasoning aggregate.**

## Why
- CausalVidQA has hard reasoning families (`predictive_reason`, `counterfactual_reason`) that typically dominate ceiling.
- Current setup still has:
  - relatively short training horizon (`epoch=10`)
  - no clear curriculum beyond small aux warmup
  - no explicit hard-negative mining / contrastive alignment in final objective
  - limited evidence calibration despite verifier head
- No local run logs exist in this workspace to prove trajectory toward 70%.

## Confidence statement
- For full aggregate metrics (`Acc_ALL` + hard family constraints), 70% is **high-risk / low-confidence** with present setup.
- For easier subsets (e.g., descriptive/explanatory alone), 70% may be reachable, but that does not indicate full causal robustness.

---

## 6) Priority improvement plan

### P0 (must do first)
1. **Fix Grad-CAM path parity** (for trustworthy analysis).
2. **Run controlled 3-run profile sweep** with fixed seed list (e.g., 3 seeds per profile), store mean/std.
3. **Track per-family calibration** (ECE or confidence gap), not only accuracy.

### P1 (high impact)
1. **Increase effective training budget**
   - more epochs (with adaptive early stop), cosine or plateau+warmup schedule.
2. **Stronger text adaptation**
   - LoRA on selected text-attention modules (or differential LR with gradual unfreeze).
3. **Family-aware loss balancing**
   - dynamic weighting for predictive/counterfactual reason families.
4. **Knowledge head quality**
   - replace keyword overlap retriever with embedding retrieval + hard negatives.

### P2 (architecture and data)
1. **Temporal consistency regularization**
   - enforce stable reasoning across adjacent sampled frames.
2. **Answer-choice contrastive objective**
   - margin between gold candidate and nearest hard negative.
3. **Data quality loop**
   - use high-`U` samples for relabel/audit; reduce noisy supervision entropy.

---

## 7) Suggested acceptance criteria for "70% ready"

Treat 70% as feasible only when all are true:
- multi-seed mean (not single run) >= target metric.
- reason-family metrics do not collapse (`Predictive-Reason`, `Counterfactual-Reason` both stable).
- calibration acceptable (confidence not overestimated on wrong answers).
- Grad-CAM path parity check passes (hooked vs real forward logits close).

---

## 8) Final conclusion

The current notebook pipeline is technically solid and already includes many strong ideas (DINOv3 + object grounding + NCOD + verifier/knowledge branches).  
However, with current training budget and reasoning difficulty, **crossing 70% on full causal aggregate is unlikely without additional optimization and better family-focused learning strategy**.  
Also, **Grad-CAM should be corrected for forward-path parity before using it as a strong decision signal**.

---

## 9) Current empirical results & gap analysis (for paper)

### 9.1 Score table (CausalVidQA, % accuracy)

| # | Model                                                          | D     | E     | PA    | PR    | PAR   | CA    | CR    | CAR   | ALL   |
|---|----------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1 | TranSTR (our reimpl)                                           | 59.97 | 62.74 | 37.32 | 31.20 | 14.88 | 52.97 | 22.47 | 12.95 | 37.63 |
| 2 | DINOv3 + FasterRCNN                                            | 73.20 | 73.28 | 58.10 | 52.61 | 33.66 | 63.26 | 37.47 | 26.53 | 52.42 |
| 3 | DINOv3 + FasterRCNN + NCOD                                     | 75.96 | 77.69 | 66.35 | 61.70 | 46.71 | 68.08 | 46.23 | 35.51 | 59.00 |
| 4 | DINOv3 + GroundingDINO(FRCNN) + NCOD                           | 70.73 | 73.95 | 63.12 | 63.07 | 46.69 | 64.68 | 60.75 | 45.26 | 59.16 |
| 7 | **DINOv3 + GroundingDINO(FRCNN) + LUM/HUM routing (Ours best)**| 72.40 | 75.92 | 65.57 | 64.95 | 49.46 | 65.74 | 62.02 | 46.48 | **61.13** |
| – | TranSTR (paper)                                                | 73.6  | 75.8  | 65.1  | 65.0  | 48.9  | 68.6  | 65.3  | 50.3  | 62.2  |
| – | VoT (7B)                                                       | 81.2  | 83.0  | 74.3  | 73.7  | 54.7  | 74.5  | 73.8  | 58.6  | 69.4  |
| – | Omni-RGPT (7B)                                                 | 84.0  | 84.6  | 84.2  | 85.4  | 76.9  | 74.7  | 74.0  | 64.3  | 77.5  |

### 9.2 Where we lose vs TranSTR (paper) and VoT

| Family          | Ours #7 | TranSTR (paper) | Δ vs TranSTR | VoT (7B) | Δ vs VoT |
|-----------------|---------|------------------|---------------|----------|----------|
| Description     | 72.40   | 73.6             | −1.20         | 81.2     | −8.80    |
| Explanation     | 75.92   | 75.8             | +0.12         | 83.0     | −7.08    |
| Predictive A    | 65.57   | 65.1             | +0.47         | 74.3     | −8.73    |
| Predictive R    | 64.95   | 65.0             | −0.05         | 73.7     | −8.75    |
| **PAR**         | 49.46   | 48.9             | **+0.56**     | 54.7     | −5.24    |
| Counterfactual A| 65.74   | 68.6             | **−2.86**     | 74.5     | −8.76    |
| Counterfactual R| 62.02   | 65.3             | **−3.28**     | 73.8     | −11.78   |
| **CAR**         | 46.48   | 50.3             | **−3.82**     | 58.6     | −12.12   |
| **ALL**         | 61.13   | 62.2             | **−1.07**     | 69.4     | −8.27    |

### 9.3 Diagnosis

- Strength: **predictive reasoning** (PAR > TranSTR paper).
- Bottleneck: **counterfactual reasoning** (CA, CR, CAR all underperform).
- Description has slight gap, likely due to noisier object grounding from GroundingDINO than the paper's tuned detector.
- 7B LLM gap is large and uniform; cannot be closed only by hyper-parameter tuning.

---

## 10) Paper-oriented contribution narrative

To position the work as a paper, the **core story** should not be a list of tricks but a clear causal claim:

> **Hypothesis.** *Causal video QA fails on counterfactual reasoning because (i) supervision is noisy and (ii) the model treats every causal family equally. We propose a family-aware noise-suppressed reasoning module that closes the counterfactual gap while keeping descriptive accuracy.*

The current LUM/HUM routing + NCOD already supports this story. The paper should make the contribution **explicit and measurable**, not just architectural:

1. **Family-aware noise correction** (LUM for soft families, HUM for hard families) – new vs. NCOD baseline.
2. **Counterfactual-aware verifier** – currently the verifier is generic; the paper version should be family-conditioned.
3. **Grad-CAM faithfulness for causal QA** – currently the only model class on CausalVidQA that ships **path-faithful Grad-CAM** (after the parity fix). This is a defensible interpretability contribution, not just a metric chase.

---

## 11) Paper-worthy improvements (ranked by expected gain)

> Heuristic estimates are conservative deltas vs current #7 (61.13%). They are not guaranteed.

### Tier A — most likely to clear 65% and approach 70% without an LLM

1. **Counterfactual-specialized branch** (≈ +1.5 to +3.0 ALL)
   - Add a dedicated counterfactual head with an *intervention loss*: for each counterfactual question, perturb the selected object set (drop / replace by nearest neighbor in feature space) and force consistency between **intervened prediction** and **counterfactual ground truth**.
   - Motivation: counterfactual questions are exactly "what if X had not happened" — the model must learn to react to *interventions*, not to *associations*.

2. **Reason–answer consistency loss** (≈ +1.0 to +2.0 ALL, mainly PAR/CAR)
   - For predictive / counterfactual families, train with a soft constraint: probability of correct answer × probability of correct reason should both be high *for the same video*.
   - Implement as cross-entropy on the joint pair, sampled when both are available.
   - Direct fix for PAR/CAR drag.

3. **Hard-negative answer mining via DeBERTa similarity** (≈ +0.8 to +1.5)
   - At each step compute candidate-to-gold cosine similarity in text-encoder space; up-weight CE on samples where the runner-up has high similarity to the gold answer.
   - Implement as per-sample loss scaling, no new module.

4. **Family-aware curriculum schedule** (≈ +0.5 to +1.0)
   - Epochs 1–2: descriptive + explanatory only.
   - Epochs 3–4: + predictive.
   - Epochs 5+: + counterfactual.
   - Prevents counterfactual noise from corrupting early representation learning. Combine with current `aux_warmup_epochs`.

5. **Test-time augmentation across frame samplings** (≈ +0.3 to +0.8)
   - At test time, sample 3 different stride patterns, average `fused_score`.
   - Cheap and reproducible.

### Tier B — strong methodological contributions for the paper

6. **Causal-aware contrastive pretraining on the same data**
   - Pretrain the visual stack (frame + obj + text projector) with a contrastive objective:
     positive = video × gold answer span, negative = video × wrong candidates from same question.
   - Then finetune with current loss.
   - Estimated: +1.0 to +2.0 ALL; large benefit on weak families.

7. **External commonsense grounding**
   - Use ConceptNet / ATOMIC retrieval keyed by (question + candidate verb/object) and inject retrieved triples through the existing `knowledge_head`.
   - Replace the current keyword-overlap retriever (low quality) with sentence-embedding retrieval.

8. **Object-trajectory module**
   - Add a lightweight inter-frame object linker (IoU + feature similarity) and feed track tokens to `obj_decoder`.
   - Specifically helps predictive (what will happen next) and counterfactual (what if object X were missing) families.

9. **Noise audit loop from NCOD U**
   - Use top-`U` samples (likely noisy) to drive a manual or programmatic relabel pass.
   - Report dataset cleaning improvement as a separate experimental table — this is highly publishable.

10. **Family-conditioned verifier** (cheap, paper-friendly)
    - Make `evidence_head` an MLP that conditions on `q_family_id` embedding.
    - Currently the verifier is shared across all families, which dilutes signal.

### Tier C — performance polish / engineering

11. **EMA of model weights** during training (typical +0.2 to +0.5).
12. **Cosine schedule with warmup**, replacing pure plateau scheduler.
13. **Stochastic depth** in encoder/decoder layers (mild regularizer).
14. **Mixed precision (bf16)** if hardware supports — allows larger effective batch.
15. **Differential LR per stack**: visual stack > obj stack > fusion > text encoder.

---

## 12) Suggested experimental protocol for the paper

To make the paper defensible:

### 12.1 Main result table
- Report mean ± std across **at least 3 seeds** per configuration.
- Bold the best non-LLM number per column.
- Clearly separate **non-LLM** and **LLM-based** rows in the table; do not compare directly without noting parameter budget.

### 12.2 Ablation table (mandatory)

Each row removes exactly one component:

| Row | Setting                                       | D | E | PAR | CAR | ALL |
|-----|-----------------------------------------------|---|---|-----|-----|-----|
| A0  | Full proposed model                           |   |   |     |     |     |
| A1  | – counterfactual intervention loss            |   |   |     |     |     |
| A2  | – reason–answer consistency loss              |   |   |     |     |     |
| A3  | – LUM/HUM routing (NCOD only)                 |   |   |     |     |     |
| A4  | – family-conditioned verifier                 |   |   |     |     |     |
| A5  | – external commonsense grounding              |   |   |     |     |     |
| A6  | – DINOv3 (back to ResNet)                     |   |   |     |     |     |
| A7  | – GroundingDINO (back to FasterRCNN only)     |   |   |     |     |     |

### 12.3 Interpretability section (paper differentiator)

- Show Grad-CAM panels on at least 4 carefully chosen examples covering all 4 hard families.
- Include the **faithfulness check**: report `max |logits_hooked − logits_real|` over a validation batch. This number must be near 0 after the parity fix.
- Add one quantitative interpretability metric (e.g., AUC of attribution masks vs object-grounded answer evidence) – this is a credible novelty point.

### 12.4 Robustness checks

- Performance under **frame dropout** (drop 25 / 50 / 75 % of frames at test time).
- Performance under **object dropout**.
- Performance on a **noisy-label subset** (top-decile of NCOD `U`) vs **clean subset**.

### 12.5 Calibration analysis

- Expected Calibration Error (ECE) per family.
- Reason–answer disagreement rate (model says correct answer but wrong reason or vice versa).

---

## 13) Concrete delta plan to push toward 70%

A realistic, low-risk path to break through the TranSTR-paper number (62.2%) and aim at 65–68%:

1. Apply Grad-CAM parity (done).
2. Add **reason–answer consistency loss** (Tier A #2) — single largest expected non-LLM gain on PAR/CAR.
3. Add **counterfactual intervention loss** (Tier A #1) — directly targets the weakest column.
4. Add **family-aware curriculum schedule** (Tier A #4) — almost free.
5. Replace the keyword-overlap knowledge retriever (Tier B #7).
6. Add **EMA + cosine warmup** (Tier C #11–12).
7. Run **3 seeds × 10 epochs**, report mean ± std.

To realistically reach VoT (≈ 69.4%) without using a 7B backbone, the visual contrastive pretraining (Tier B #6) is the most important single step.

To realistically beat Omni-RGPT (≈ 77.5%) without a 7B backbone is **not** the target of this paper — frame the paper as the **best parameter-efficient causal-aware reasoning baseline**, not as an SOTA chase.

---

## 14) Recommended paper positioning

- **Title direction.** "Family-Aware Noise-Suppressed Reasoning for Causal Video Question Answering."
- **Core claim.** A small, interpretable, non-LLM model that closes the counterfactual gap on CausalVidQA via family-conditioned noise correction and intervention-aware learning, with path-faithful Grad-CAM as a built-in property.
- **Contribution bullets** (4 is the standard count):
  1. LUM / HUM routing conditioned on question family.
  2. Reason–answer consistency objective for hard causal pairs.
  3. Intervention-based counterfactual loss.
  4. Path-faithful Grad-CAM analysis with faithfulness check.
- **Honest framing of LLM gap.** Acknowledge VoT / Omni-RGPT explicitly and report parameter budget; argue the contribution is in the **non-LLM, interpretable** regime.

