# Analysis Agent — PerturbVLA Diagnosis Plan
## "Closing the LIBERO-PRO Collapse: Analysis-First Approach"

**Branch:** agent/analysis-perturbvla
**Goal:** Diagnose WHY VLA models memorize trajectories instead of understanding tasks, then derive a fix from the diagnosis.

---

## Research Question
When LIBERO-PRO applies perturbations (object replacement, corrupted instructions, spatial shifts), VLA success collapses from >90% to 0%. **What component of the VLA pipeline fails?**

## Hypothesis Tree

### H1: Vision encoder ignores scene details
- The VLM backbone may rely on global scene statistics rather than object-level features
- Test: Replace vision encoder with frozen vs. fine-tuned; measure robustness delta
- Intervention: Gradually mask visual tokens and measure success degradation curve

### H2: Language conditioning is weak/ignored
- The model may learn a "language-agnostic" shortcut — same actions regardless of instruction
- Test: Provide contradictory instructions (e.g., "pick up the red cup" when only blue exists); measure if behavior changes
- Intervention: Ablate language tokens at inference; measure success delta

### H3: Action head memorizes trajectory distributions
- The action decoder may collapse to a unimodal prior over training trajectories
- Test: Measure action variance across different initial states for the same task
- Intervention: Replace action head with one trained on diverse trajectories; compare

### H4: Proprioceptive state creates a shortcut
- The model may use proprioception as a "cheat code" — if the arm is already near the object, it doesn't need vision
- Test: Randomize initial joint positions; measure success degradation
- Intervention: Ablate proprioceptive input at inference

### H5: Temporal correlation in action sequences
- Consecutive actions are highly correlated; the model may predict "the next action looks like the current one"
- Test: Insert action perturbations mid-trajectory; measure recovery
- Intervention: Action chunking with diverse sampling

---

## Diagnosis Experiments (Priority Order)

### Experiment 1: Vision-Language Attention Audit
**What:** Record attention maps from the VLM backbone during inference
**How:** Hook into the cross-attention layers; visualize what the model "looks at" when generating each action
**Expected:** If H1 is correct, attention will be diffuse/distributed, not focused on task-relevant objects
**Deliverable:** Attention heatmaps for 10 tasks × 3 conditions (clean, spatial perturbation, object replacement)

### Experiment 2: Language Ablation Study
**What:** Run inference with language token dropout (0%, 25%, 50%, 75%, 100%)
**How:** Zero out language embeddings at inference time; measure success rate curve
**Expected:** If H2 is correct, success rate will be insensitive to language dropout (flat curve)
**Deliverable:** Success rate vs. language dropout percentage for each task suite

### Experiment 3: Action Diversity Analysis
**What:** Measure action distribution entropy across episodes for the same task
**How:** Run 50 episodes per task; compute per-step action variance and KL divergence from uniform
**Expected:** If H3 is correct, action variance will be very low (memorized single trajectory)
**Deliverable:** Action entropy histograms per task; comparison across clean vs. perturbed conditions

### Experiment 4: Component Isolation
**What:** Systematically ablate each input modality at inference
**How:** Run inference with: (a) vision only, (b) language only, (c) proprioception only, (d) all combined
**Expected:** Reveals which modality the model actually depends on
**Deliverable:** Success rate matrix: modality × task suite

### Experiment 5: Perturbation Sensitivity Map
**What:** Apply perturbations to individual components and measure success delta
**How:** For each perturbation type (spatial, visual, language, temporal), apply to: vision encoder input, language input, proprioceptive input, action output
**Expected:** Reveals the "weakest link" — which component's perturbation causes the largest drop
**Deliverable:** 4×4 perturbation sensitivity matrix

---

## From Diagnosis to Fix

After experiments 1-5, we will know:
1. Which component fails first under perturbation
2. Whether the failure is in perception (H1, H2) or action (H3, H5)
3. Whether the model uses shortcuts (H2, H4)

The fix will be **derived from the diagnosis**, not prescribed in advance. Possible fixes include:
- If H1: Adversarial vision augmentation during training (not at inference)
- If H2: Language-conditioned contrastive loss
- If H3: Action distribution regularization
- If H4: Proprioception dropout during training
- If H5: Action sequence diversity augmentation

**Key principle:** The analysis paper's contribution is the DIAGNOSIS, not the fix. The fix is validation that the diagnosis was correct.

---

## Deliverables Timeline

| Week | Deliverable | GPU Hours |
|------|-------------|-----------|
| 1 | Exp 1+2: Attention audit + Language ablation | ~10h |
| 2 | Exp 3+4: Action diversity + Component isolation | ~15h |
| 3 | Exp 5: Perturbation sensitivity map | ~10h |
| 4 | Analysis write-up + Fix design | 0h (CPU only) |
| 5-6 | Fix implementation + Ablation validation | ~30h |
| 7-8 | Real-robot validation (SO-101) | ~20h |

**Total GPU budget:** ~85 hours on L4 (fits within remaining AWS budget)

---

## Baseline Reproduction (MANDATORY)

Before any diagnosis experiments, reproduce:
1. **RobustVLA** (2025) — their numbers on LIBERO-PRO
2. **RoVLA** (2026) — their numbers on LIBERO-Plus

These are the baselines we must beat. If we can't reproduce them, we can't claim improvement.

---

## Status: 🟡 PLAN COMPLETE — awaiting GPU environment setup
