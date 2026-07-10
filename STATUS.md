# Agent Status Table
*Last updated: 2026-07-10 10:55 GMT+8*

| Agent | Job | Branch | Status | GPU | Last Checkpoint | ETA |
|-------|-----|--------|--------|-----|-----------------|-----|
| **Skeptic** | Audit fabricated results | main (merged) | ✅ DONE | — | — | — |
| **Skeptic** | Paper \TODO{} scrubbing | main (merged) | ✅ DONE | — | — | — |
| **Analysis** | PerturbVLA diagnosis plan | agent/analysis-perturbvla | ✅ PLAN DONE | — | — | — |
| **Analysis** | Baseline reproduction (RobustVLA, RoVLA) | agent/analysis-perturbvla | ⏳ BLOCKED | L4 (24GB) | — | Awaiting env |
| **Analysis** | Exp 1-5: Diagnosis experiments | agent/analysis-perturbvla | ⏳ BLOCKED | L4 (24GB) | — | After baselines |
| **Audit** | MultiRes-Action design-space review | agent/audit-multires | ✅ ABANDONED | — | — | — |
| **Audit** | Defensibility verdict | agent/audit-multires | ❌ ABANDON (scooped by HARP-VLA, Moto, LAPA) | — | — | — |
| **Trainer** | MambaFlow training | trainer/mambaflow | ⏳ DEPRIORITIZED | L4 (24GB) | — | Gated on WS1+WS2 |
| **Trainer** | PerturbVLA training | trainer/perturb | ⏳ REPOSITIONING | L4 (24GB) | — | After diagnosis |
| **Trainer** | MultiRes-Action training | trainer/multires | ❌ ABANDONED | — | — | Fold into PerturbVLA if diagnosed |

## Environment Status
| Resource | Status | Details |
|----------|--------|---------|
| g6.xlarge (L4, 24GB) | ✅ ACCESSIBLE | 44.234.88.211, PyTorch 2.5.1+cu121 installed |
| g6e.xlarge (L40S, 48GB) | ❌ KEY MISMATCH | 52.41.32.127, needs "gpu-key" PEM (not available) |
| mamba-ssm | ❌ BUILD FAILED | CUDA build issue on L4; pure-PyTorch fallback available |

## Completed Actions
1. ✅ Skeptic audit: fake results renamed to SIMULATED_* (commit 7e5af7a)
2. ✅ Paper scrubbing: 19 \TODO{} placeholders (commit 9a2e1b1)
3. ✅ Analysis plan: 5 diagnosis experiments (commit bdf4e07)
4. ✅ MultiRes audit: design-space matrix filled (commit 35ba822)
5. ✅ Directive v2 plan documented (DIRECTIVE_V2.md)

## Next Actions (Priority Order)
1. **Fix g6e.xlarge SSH access** — need "gpu-key" PEM or create new key pair
2. **Analysis Agent: Run baseline reproduction** — RobustVLA + RoVLA on LIBERO-PRO
3. **Audit Agent: Final defensibility verdict** — PROCEED/PIVOT/ABANDON for MultiRes-Action
4. **If MultiRes proceeds:** Design experiments to demonstrate difference from RT-H/GR00T N1
