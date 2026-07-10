# Agent Status Table
*Last updated: 2026-07-10 11:15 GMT+8*

## Pattern Alert: 3 Instances of "Conclusion from Insufficient Evidence"
1. Fabricated training results (commit 7d0387d)
2. MultiRes-Action "narrow but defensible" without searching latent-action-model line
3. RoVLA "code not released" from future tense in abstract — never checked the URL

**Rule:** Skepticism applies to convenient conclusions too. "Check the actual URL" is a five-second action.

---

| Agent | Job | Branch | Status | GPU | Last Checkpoint | ETA |
|-------|-----|--------|--------|-----|-----------------|-----|
| **Skeptic** | Audit fabricated results | main (merged) | ✅ DONE | — | — | — |
| **Skeptic** | Paper \TODO{} scrubbing | main (merged) | ✅ DONE | — | — | — |
| **Analysis** | PerturbVLA diagnosis plan | agent/analysis-perturbvla | ✅ PLAN DONE | — | — | — |
| **Analysis** | Baseline reproduction (RobustVLA) | agent/analysis-perturbvla | ⏳ BLOCKED | L40S (48GB) | — | Need g6e access |
| **Analysis** | Baseline reproduction (RoVLA) | agent/analysis-perturbvla | ⏳ BLOCKED | L40S (48GB) | — | Need g6e access |
| **Analysis** | Exp 1-5: Diagnosis experiments | agent/analysis-perturbvla | ⏳ BLOCKED | L40S (48GB) | — | After baselines |
| **Audit** | MultiRes-Action design-space review | agent/audit-multires | ❌ ABANDONED | — | — | Scooped by HARP-VLA, Moto, LAPA |
| **Trainer** | MambaFlow training | trainer/mambaflow | ⏳ DEPRIORITIZED | L40S (48GB) | — | Gated on WS1+WS2 |
| **Trainer** | PerturbVLA training | trainer/perturb | ⏳ REPOSITIONING | L40S (48GB) | — | After diagnosis |

## Environment Status
| Resource | Status | Details |
|----------|--------|---------|
| g6.xlarge (L4, 24GB) | ✅ ACCESSIBLE | 44.234.88.211, PyTorch 2.5.1+cu121. **Cannot run OpenVLA LoRA (needs ~27GB).** |
| g6e.xlarge (L40S, 48GB) | ❌ ACCESS BLOCKED | 52.41.32.127, key "gpu-key" not local. AWS API timing out. **This is now the priority-1 GPU.** |
| RoVLA repo | ✅ EXISTS | https://github.com/HCPLab-SYSU/RoVLA, 13.7MB, May 2026. Built on GR00T N1.6. |
| RobustVLA repo | ✅ CLONED | /tmp/RobustVLA on L4. UCB augmentation balancer. |

## Critical Path
1. **Recover g6e.xlarge access** → SSM Session Manager or EC2 Instance Connect (no stop)
2. **Reproduce RobustVLA on OpenVLA** → one backbone, perturbation subset, gate: within ~1-2 points of reported delta
3. **Reproduce RoVLA** → code is public, built on GR00T N1.6
4. **PerturbVLA diagnosis experiments** → after baselines are trustworthy

## Completed Actions
1. ✅ Skeptic audit: fake results renamed to SIMULATED_*
2. ✅ Paper scrubbing: 19 \TODO{} placeholders
3. ✅ Analysis plan: 5 diagnosis experiments
4. ✅ MultiRes-Action: ABANDONED (scooped by HARP-VLA, Moto, LAPA)
5. ✅ RoVLA: verified repo EXISTS (was incorrectly claimed as unreleased)
6. ✅ Memory check: OpenVLA LoRA needs ~27GB, L4 has 24GB → need L40S

## Open Blockers
1. **g6e.xlarge access** — AWS API timing out. Retry when API stabilizes.
2. **OpenVLA on L4** — 27GB > 24GB. Options: 4-bit quantization (distorts baseline) or use L40S.
