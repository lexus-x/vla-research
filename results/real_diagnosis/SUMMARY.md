# Diagnosis Results — OpenVLA finetuned-libero-spatial on L4

## Setup
- Model: openvla/openvla-7b-finetuned-libero-spatial
- GPU: NVIDIA L4 (24GB), 15.10GB VRAM used
- Inference: ~483ms/step (after warmup)
- Framework: transformers 4.46.3, torch 2.5.1+cu121, timm 0.9.16

## EXP1: Visual Perturbation
**Finding:** Actions change meaningfully under visual perturbation.
- Action std ranges 0.08-0.73across dimensions
- Gaussian blur perturbation causes consistent action variation
- Latency stable at ~483ms

## EXP2: Language Ablation
**Finding:** Language DOES affect actions in finetuned model (unlike base OpenVLA).
- Base OpenVLA: byte-identical actions at 0-75% dropout (from commit 775334e)
- Finetuned OpenVLA: actions diverge at 75%+ dropout
- Key: gripper dimension changes 0.664→ 0.996 at100% dropout
- Subtle effect: 0-50% dropout produces nearly identical actions
- **Interpretation:** LIBERO finetuning teaches the model to use language, primarily for gripper control

## EXP3: Temporal Consistency
**Finding:** Actions are perfectly deterministic.
- Variance =0.0 across5repeated inferences
- Same input always produces identical output

## Implications for PerturbVLA
1. Finetuned VLAs DO use language (unlike base models) — language perturbation during training is meaningful
2. Visual perturbation causes real action changes — visual robustness training is justified
3. Deterministic actions mean perturbation effects are systematic, not random noise
4. The ~483ms inference latency is within the range for real-time control (2Hz)

## Next Steps
- Run with real LIBERO episodes (not synthetic images)
- Test with all4LIBERO suites (spatial, object, goal, long)
- Measure success rate under perturbation (not just action change)
- Compare with RobustVLA baseline
