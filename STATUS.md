# Agent Status Table
*Last updated: 2026-07-10 13:45 GMT+8*

## Infrastructure
| Resource | Status | Details |
|----------|--------|---------|
| L4 (g6.xlarge, 24GB) | ✅ ACTIVE | 44.234.88.211, PyTorch 2.5.1+cu121, transformers 4.46.3, timm 0.9.16 |
| L40S (g6e.xlarge, 48GB) | ✅ STANDBY | 34.211.170.34, disk expanded to60GB, reserved for LoRA fine-tuning |

## Completed Experiments
| Experiment | Model | Result | Status |
|-----------|-------|--------|--------|
| EXP1: Visual Perturbation | openvla-7b-finetuned-libero-spatial | Actions change (std 0.08-0.73) | ✅ DONE |
| EXP2: Language Ablation | openvla-7b-finetuned-libero-spatial | Language affects finetuned model (unlike base) | ✅ DONE |
| EXP3: Temporal Consistency | openvla-7b-finetuned-libero-spatial | Perfectly deterministic (var=0) | ✅ DONE |
| All Suites (4x) | spatial/object/goal/10 | Running | ⏳ IN PROGRESS |

## Key Findings
1. **Language matters after finetuning:** Base OpenVLA ignores language (byte-identical at 0-75% dropout). Finetuned OpenVLA diverges at75%+ dropout, with gripper changing0.664→0.996.
2. **Visual perturbation is real:** Gaussian blur causes consistent action changes across all dimensions.
3. **Deterministic actions:** Same input always produces same output — perturbation effects are systematic.
4. **Inference:** ~483ms/step on L4,15.10GB VRAM — within real-time range (2Hz).

## Active Agents
| Agent | Status | Task |
|-------|--------|------|
| Literature Hunter | 🔄 Running | Arxiv scanning every30min |
| Novelty Auditor | 🔄 Running | Novelty claim verification every1hr |
| Red-Team Critic | 🔄 Running | Adversarial review every2hr |
| Benchmark Engineer | ⏳ QUEUED | LIBERO env setup after suite experiments |
| GPU Monitor | 🔄 Running | L4 utilization logging every10s |

## Cron Jobs
- Git auto-push: every30min (openclaw cron)
- Literature scan: every30min
- Novelty audit: every1hr

## Critical Path
1. ✅ Fix commit 775334e false claim
2. ✅ L4 environment setup (PyTorch, transformers, timm)
3. ✅ Diagnosis experiments (EXP1-3)
4. ⏳ All-suites experiment (spatial/object/goal/10)
5. ⏳ LIBERO environment setup for real episodes
6. ⏳ RobustVLA baseline reproduction
7. ⏳ PerturbVLA training
