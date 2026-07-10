# VLA Novel Breakthrough Research

PhD-level research project targeting novel breakthroughs in Vision-Language-Action (VLA) models.

## Top 3 Ideas (Novelty-Verified)

### 1. PerturbVLA — Adversarial Perturbation Training (Score: 9.6)
- **Problem:** VLAs memorize trajectories (LIBERO-PRO: 90% → 0% under perturbation)
- **Solution:** Systematic adversarial perturbation training + contrastive robustness loss
- **Size:** 0 extra params (training method)
- **Expected:** Maintain >80% SR under perturbation

### 2. MultiRes-Action — Coarse-to-Fine Action Head (Score: 9.4)
- **Problem:** Long-horizon tasks are hardest (5-15% gap)
- **Solution:** Hierarchical coarse planner (flow matching) + fine controller (MLP)
- **Size:** <10M params (plug-in module)
- **Expected:** +10-15% on long-horizon tasks

### 3. MambaFlow — SSM Backbone + Flow Matching (Score: 9.0)
- **Problem:** Transformer VLAs too slow for real-time control (200-500ms)
- **Solution:** Mamba (SSM) backbone with flow matching action head
- **Size:** 300-500M params
- **Expected:** 3-5x faster inference

## Benchmarks
- LIBERO, LIBERO+, MetaWorld (simulation only)

## Hardware
- AWS g6.2xlarge (L40S, 48GB VRAM)

## Research Pipeline
1. ✅ Literature survey (25 models, 1490+ lines)
2. ✅ Cross-domain mining (15 techniques)
3. ✅ Architecture decomposition (14 models)
4. ✅ Benchmark analysis
5. ✅ Novelty verification (55+ papers)
6. ✅ Top 3 selection and gate evaluation
7. ✅ Implementation (code/perturb_vla, code/multires_action, code/mambaflow)
8. ⏳ Training on AWS
9. ⏳ Evaluation and ablation
10. ⏳ Paper writing

