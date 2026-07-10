# VLA Research Constraint Gates

## Gate 1: NOVELTY (Must Pass)
- [ ] Searched arXiv, Semantic Scholar, Google Scholar for the core idea
- [ ] No existing paper claims the same approach for VLA
- [ ] If cross-domain: no paper has transferred this technique to VLA before
- [ ] Novelty score: **Must be ≥ 8/10** (10 = completely new, 1 = minor variation)
- [ ] Documented in `05_novelty_database.md` with evidence

## Gate 2: FEASIBILITY (Must Pass)
- [ ] Total model size: <500M params (if standalone) OR module size: <10M params (if plug-in)
- [ ] Target: **10x or >10x smaller than baseline** (e.g., OpenVLA 7B → <700M)
- [ ] Can be trained on 1x L40S (48GB VRAM) within reasonable time (<24h)
- [ ] Uses simulation only: LIBERO, LIBERO+, MetaWorld
- [ ] No real-world hardware required

## Gate 3: PERFORMANCE (Must Pass)
- [ ] Expected improvement ≥5% over baseline on at least 2 metrics
- [ ] Baselines to beat:
  - OpenVLA (7B) on LIBERO
  - π₀ on LIBERO
  - Octo on LIBERO
  - Any SOTA on MetaWorld
- [ ] Statistical significance: 3 seeds minimum, report mean ± std
- [ ] Improvement in at least TWO of:
  - Success rate (SR)
  - Inference speed (Hz / ms per step)
  - Model size (params / MB)
  - Robustness (cross-task generalization)
  - Sample efficiency (fewer training steps/data)

## Gate 4: QUALITY (Must Pass for Q1/Q2 Journal)
- [ ] Novelty is significant enough for a top venue (not incremental)
- [ ] Rigorous experimental design:
  - Controlled baselines (same data, same compute budget)
  - Ablation study on each proposed component
  - Statistical reporting (mean, std, confidence intervals)
  - Multiple benchmark suites
- [ ] Clear problem statement and motivation
- [ ] Thorough related work (covering 50+ papers)
- [ ] Reproducible code + configs

## Gate 5: IMPACT (Scoring)
- Speed improvement: ___x faster than baseline
- Size reduction: ___x smaller than baseline
- Success rate improvement: +___% over baseline
- Robustness improvement: +___% on cross-task/distribution shift
- **Minimum: Top 2 of these must show improvement**

## GATE EVALUATION SCORECARD

| Criterion | Weight | Min Score | Actual |
|-----------|--------|-----------|--------|
| Novelty | 30% | 8/10 | ___ |
| Performance Gain | 25% | ≥5% SR improvement | ___ |
| Size Efficiency | 20% | 10x smaller than 7B | ___ |
| Speed | 15% | ≥2x faster inference | ___ |
| Reproducibility | 10% | Full code + configs | ___ |

**PASS THRESHOLD: Weighted score ≥ 7.5/10**

## TOP 3 SELECTION CRITERIA
After filtering through all gates, rank by:
1. **Performance gain × Novelty** (weighted 60/40)
2. Feasibility confidence (can we actually do it in time?)
3. Potential for follow-up work (extends to a research line)

Select TOP 3 → implement in parallel on AWS + local.
