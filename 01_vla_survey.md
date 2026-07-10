# Comprehensive Survey: Vision-Language-Action (VLA) Models (2024–2026)

*Last updated: 2026-07-10*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Model Catalog](#model-catalog)
3. [Architecture Taxonomy](#architecture-taxonomy)
4. [Benchmark Comparison](#benchmark-comparison)
5. [Limitations & Open Problems](#limitations--open-problems)
6. [References](#references)

---

## Introduction

Vision-Language-Action (VLA) models are end-to-end multimodal foundation models that map visual observations and language instructions directly to robot actions. They emerged from the convergence of large vision-language models (VLMs) and robot learning, promising generalist robot policies that can follow natural-language instructions across diverse tasks and embodiments.

This survey covers all major VLA architectures from 2024 to mid-2026.

---

## Model Catalog

### RT-1 (Google, 2022)
- **Venue:** RSS 2023 (arXiv 2022)
- **Architecture:** EfficientNet + TokenLearner + Transformer
- **Parameters:** 35M
- **Action Head:** Discrete tokens (autoregressive)
- **Training Data:** 130K demonstrations from Google robot fleet
- **Key Innovation:** First large-scale robot transformer policy; demonstrated that transformer architectures can effectively learn robot control from demonstrations
- **Key Results:** 97% on in-distribution tasks; significant drop on out-of-distribution
- **Limitations:** Single robot embodiment; poor OOD generalization; no language grounding

---

### RT-2 (Google DeepMind, 2023)
- **Venue:** arXiv 2023
- **Architecture:** Built on PaLI-X (55B) or PaLM-E (12B); ViT-e (4B) vision encoder; actions as text strings of discretized joint positions
- **Parameters:** 55B (PaLI-X variant) or 12B (PaLM-E variant)
- **Action Head:** Discrete tokens — actions represented as text strings (e.g., "1 128 91 241 5 101 127") and generated autoregressively
- **Training Data:** ~130K demonstrations from 13 robot types (RT-X dataset) + internet-scale VL pretraining
- **Key Innovation:** First proof that VLA concept works at scale; co-fine-tuning robot action data with internet pretraining data to preserve world knowledge; emergent capabilities (e.g., following instructions about never-seen concepts like "move to Taylor Swift album")
- **Key Results:** 62% success on novel semantic concepts (vs 32% for RT-1); strong zero-shot generalization
- **Limitations:** Closed source; massive compute required; 55B parameters make real-time inference challenging; single-step discrete actions produce jerky motion

---

### Octo (UC Berkeley, 2024)
- **Venue:** arXiv 2024
- **Architecture:** Transformer backbone trained from scratch (no pretrained LLM); diffusion action head; optional language conditioning via pretrained text encoder
- **Parameters:** ~93M
- **Action Head:** Diffusion (DDPM-style iterative denoising)
- **Training Data:** Open X-Embodiment (~800K trajectories)
- **Key Innovation:** Designed specifically for cross-embodiment transfer and rapid fine-tuning; smallest generalist robot policy; diffusion head handles multi-modal actions
- **Key Results:** 50–75% success on SIMPLER benchmark; fine-tuning on 20–50 demos takes 30 min–2 hours
- **Limitations:** No internet-scale pretraining → not a true VLA by strict definition; limited language understanding; lower performance than larger VLAs on complex tasks

---

### OpenVLA (Stanford/Berkeley/TRI, 2024)
- **Venue:** arXiv 2024 (June)
- **Architecture:** Prismatic VLM backbone (Llama-2 7B + SigLIP + DinoV2 dual vision encoders); projector maps visual embeddings into LLM space
- **Parameters:** 7.5B
- **Action Head:** Discrete tokens — actions discretized into 256 bins per dimension, predicted autoregressively
- **Training Data:** 970K robot episodes from Open X-Embodiment (22 robot embodiments); trained on 64 A100 GPUs for 15 days
- **Key Innovation:** First fully open-source VLA at scale; dual vision encoder (SigLIP semantic + DinoV2 spatial); outperforms RT-2-X (55B) despite being 7× smaller; LoRA fine-tuning with only 1.4% parameters
- **Key Results:** SOTA generalist robot manipulation; 85–95% on LIBERO (fine-tuned); outperforms RT-1-X, Octo, and matches/exceeds RT-2-X; strong on visual/motion/physical/semantic generalization
- **Limitations:** ~300ms per action step (too slow for reactive tasks); discrete tokens produce less smooth actions; requires significant VRAM for fine-tuning (2× A100 for full, 1× A100 for LoRA); struggles with novel internet concepts vs RT-2-X

---

### π₀ (Physical Intelligence, 2024)
- **Venue:** arXiv 2024 (October); blog post October 31, 2024
- **Architecture:** 3B parameter pretrained VLM backbone + novel "action expert" module using flow matching; flow matching generates actions by iteratively transforming noise into action trajectories via learned velocity field
- **Parameters:** ~3B
- **Action Head:** Flow matching (variant of diffusion) — produces continuous action outputs at up to 50 Hz; augments pretrained VLM with continuous action outputs
- **Training Data:** Largest robot interaction dataset at the time: Open X-Embodiment + proprietary π dataset across 8 distinct robots (UR5e, Bimanual UR5e, Franka, Bimanual Trossen, Bimanual Arx, Mobile Trossen, Mobile Fibocom) + internet-scale VL pretraining
- **Key Innovation:** Flow matching for smooth continuous action generation; cross-embodiment training mixture; post-training (fine-tuning) analogous to LLM post-training for dexterous tasks; demonstrated emergent strategies (stacking dishes, recovering from intervention)
- **Key Results:** Outperforms OpenVLA (7B) and Octo (93M) on complex multi-stage tasks; demonstrated laundry folding, table bussing, box assembly, grocery packing; 80–95% on trained tasks; 40–60% zero-shot on related unseen tasks
- **Limitations:** Closed source (weights now partially released via openpi); 10–20 denoising steps add latency; proprietary training data; tasks evaluated are harder than academic benchmarks but limited in scope

---

### π₀.₅ (Physical Intelligence, 2025)
- **Venue:** arXiv / blog post April 22, 2025
- **Architecture:** Extends π₀ with co-training on heterogeneous data: multimodal web data (captioning, VQA, object detection) + robot action data from multiple environments + verbal instruction demonstrations + cross-embodiment data; high-level subtask prediction + low-level action expert
- **Parameters:** ~3B (same as π₀)
- **Action Head:** Flow matching (same as π₀)
- **Training Data:** Co-training mixture: multimodal web data, robot data from many environments (static + mobile robots), cross-embodiment data from π₀, verbal instruction demonstrations, subtask commands
- **Key Innovation:** Open-world generalization — deploys in entirely new homes not seen in training; co-training recipe with web data for semantic understanding + multi-environment robot data for physical skills; chain-of-thought style high-level reasoning + low-level action
- **Key Results:** 83% in-distribution success rate, 86% in-distribution follow rate; 94% OOD success rate and 94% OOD follow rate (with full training mixture); web data critical for OOD object recognition; demonstrated cleaning tasks in new homes
- **Limitations:** Still closed source (openpi released with partial weights); inference latency from multi-step denoising; goal is generalization not high dexterity

---

### VLANeXt (Wu et al., 2026)
- **Venue:** arXiv Feb 2026
- **Architecture:** Systematic exploration of VLA design space starting from RT-2-like baseline; unified framework with 12 key design findings; 2.5B parameter model; "soft connection" between VLM and policy module; frequency-domain action modeling
- **Parameters:** 2.5B
- **Action Head:** Action generation framed as time-series forecasting with frequency-domain modeling
- **Training Data:** Open X-Embodiment + LIBERO/LIBERO-plus for evaluation
- **Key Innovation:** 500+ ablation experiments distilling 12 key VLA design recipes; demonstrates that principled design choices beat aggressive scaling (2.5B outperforms 7B OpenVLA-OFT); video inputs fail to help action learning; proprioceptive input conditioned in VLM outperforms alternatives
- **Key Results:** SOTA on LIBERO and LIBERO-plus benchmarks; outperforms OpenVLA-OFT (7B) with 2.5B model; strong real-world performance
- **Limitations:** Primarily evaluated on LIBERO (near-saturated benchmark); limited real-world diversity

---

### DexVLA (Wen et al., 2025)
- **Venue:** CoRL 2025 (accepted)
- **Architecture:** VLM backbone + plug-in 1B-parameter diffusion-based action expert; embodiment curriculum learning strategy (3-stage: pre-train diffusion expert on cross-embodiment data → align VLA to specific embodiments → post-train for new tasks)
- **Parameters:** ~1B diffusion expert + VLM backbone
- **Action Head:** Diffusion-based action expert (1B params), designed for cross-embodiment learning; separable from VLA
- **Training Data:** Cross-embodiment data from multiple robot types (single-arm, bimanual, dexterous hand)
- **Key Innovation:** Plug-in diffusion expert that can be pre-trained independently and attached to any VLM; embodiment curriculum learning; scales to dexterous hand manipulation
- **Key Results:** Outperforms Octo, OpenVLA, and Diffusion Policy across single-arm, bimanual, and dexterous hand tasks; demonstrated laundry folding via language prompting; efficient adaptation with limited data
- **Limitations:** Additional complexity of separate diffusion expert; training pipeline requires 3 stages

---

### CoT-VLA (NVIDIA, 2025)
- **Venue:** arXiv 2025
- **Architecture:** VLM backbone with visual chain-of-thought reasoning; generates intermediate reasoning tokens before action prediction
- **Parameters:** Not specified (likely 3-7B range)
- **Action Head:** Autoregressive with CoT reasoning tokens
- **Training Data:** Open X-Embodiment + proprietary data
- **Key Innovation:** Visual chain-of-thought reasoning for VLA — model reasons about visual scene before generating actions; improves complex multi-step tasks
- **Key Results:** Best or competitive performance across all LIBERO benchmark suites
- **Limitations:** Autoregressive CoT adds inference latency; reasoning quality depends on training data diversity

---

### Xiaomi-Robotics-0 (Xiaomi, 2026)
- **Venue:** arXiv Feb 2026; open-sourced February 2026
- **Architecture:** Pretrained VLM backbone + diffusion transformer for action generation via flow-matching; Λ-shape attention mask during post-training to prevent action-prefix shortcut learning
- **Parameters:** 4.7B
- **Action Head:** Diffusion transformer with flow-matching; chunk-based action generation with careful timestep alignment for seamless real-time rollouts
- **Training Data:** Large-scale cross-embodiment robot trajectories + vision-language data (preserves VL capabilities)
- **Key Innovation:** Asynchronous execution techniques for real-time robot deployment; Λ-shape attention mask prevents action shortcut learning; preserves VLM capabilities after robot training; runs on consumer-grade GPU
- **Key Results:** 98.7% average on LIBERO; SOTA across all three simulation benchmarks (LIBERO, SIMPLER, RoboTwin); high throughput on bimanual real-robot tasks; matches underlying VLM on VL benchmarks
- **Limitations:** 4.7B parameters still requires GPU for real-time; primarily evaluated on table-top manipulation

---

### Dita (Hou et al., 2025)
- **Venue:** ICCV 2025
- **Architecture:** Diffusion Transformer (DiT) architecture for VLA; in-context conditioning — denoised actions attend to raw visual tokens from historical observations rather than fused embeddings
- **Parameters:** Not specified (Transformer-based, likely 1-3B range)
- **Action Head:** Diffusion Transformer — directly denoises continuous action sequences through unified multimodal diffusion process; models action deltas and environmental nuances via in-context conditioning
- **Training Data:** Cross-embodiment datasets across diverse camera perspectives, observation scenes, tasks, and action spaces
- **Key Innovation:** In-context conditioning for fine-grained action-observation alignment (vs. shallow network conditioning in prior work); scales diffusion action denoiser alongside Transformer; robust to environmental variances with 10-shot fine-tuning using only third-person cameras
- **Key Results:** SOTA or competitive on simulation benchmarks; robust real-world adaptation to environmental variances and long-horizon tasks via 10-shot fine-tuning
- **Limitations:** Requires multiple denoising steps; limited to third-person camera in experiments

---

### FLOWER (Reuss et al., 2025)
- **Venue:** CoRL 2025
- **Architecture:** VLM backbone with intermediate-modality fusion (prunes up to 50% of LLM layers, reallocating capacity to diffusion head) + action-specific Global-AdaLN conditioning (20% parameter reduction via modular adaptation)
- **Parameters:** 950M
- **Action Head:** Rectified flow (flow matching) with Global-AdaLN conditioning
- **Training Data:** Open X-Embodiment + diverse cross-embodiment data; pretrained in 200 H100 GPU hours
- **Key Innovation:** Most efficient flow-based VLA; intermediate-modality fusion reallocates LLM capacity to action head; Global-AdaLN conditioning for parameter efficiency; demonstrates that sub-1B VLA can match multi-billion parameter models
- **Key Results:** New SoTA 4.53 on CALVIN ABC benchmark; competitive with bigger VLAs across 190 tasks spanning 10 simulation and real-world benchmarks; robust across diverse embodiments
- **Limitations:** Still larger than SmolVLA; limited to flow-matching action head

---

### CrossFormer (Doshi et al., 2024)
- **Venue:** arXiv Aug 2024
- **Architecture:** Scalable transformer-based policy; consumes data from any embodiment without manual alignment of observation or action spaces; flexible input/output handling for heterogeneous robots
- **Parameters:** Not specified (transformer-based)
- **Action Head:** Flexible — adapts to different action spaces per embodiment
- **Training Data:** 900K trajectories across 20 different robot embodiments (largest and most diverse cross-embodiment dataset at the time)
- **Key Innovation:** First policy to handle manipulation, navigation, locomotion, and aviation in a single model; no manual alignment of observation/action spaces required; matches specialist policy performance per embodiment
- **Key Results:** Matches specialist policies on each embodiment; significantly outperforms prior cross-embodiment SOTA; demonstrated on single/dual arm, wheeled robots, quadcopters, quadrupeds
- **Limitations:** Not a VLA in the strict sense (no internet-scale VL pretraining); primarily a multimodal policy; limited language conditioning

---

### ChatVLA (Zhou et al., 2025)
- **Venue:** EMNLP 2025
- **Architecture:** VLM backbone with Mixture-of-Experts (MoE) architecture; Phased Alignment Training — incrementally integrates multimodal data after initial control mastery
- **Parameters:** Not specified
- **Action Head:** Autoregressive discrete tokens
- **Training Data:** Robot control data + multimodal understanding data (VQA, captioning)
- **Key Innovation:** Addresses "spurious forgetting" (robot training overwrites VL alignments) and "task interference" (competing control/understanding tasks degrade each other); MoE minimizes task interference; Phased Alignment Training
- **Key Results:** 6× higher performance on MMMU vs prior VLAs; 47.2% on MMStar; superior on 25 real-world robot manipulation tasks vs OpenVLA; competitive VQA performance
- **Limitations:** Added complexity from MoE and phased training; autoregressive action generation

---

### ChatVLA-2 (Zhou et al., 2025)
- **Venue:** NeurIPS 2025
- **Architecture:** Extends ChatVLA with open-world reasoning capabilities; enhanced for complex reasoning in robotic tasks
- **Parameters:** Not specified
- **Action Head:** Autoregressive with enhanced reasoning
- **Training Data:** Extended dataset with open-world reasoning tasks
- **Key Innovation:** Open-world reasoning for VLA; improved generalization to novel environments and tasks
- **Key Results:** Improved reasoning capabilities over ChatVLA; demonstrated complex multi-step reasoning in robotic tasks
- **Limitations:** Increased complexity; still autoregressive

---

### StarVLA-α (Ye et al., 2026)
- **Venue:** arXiv April 2026
- **Architecture:** Simple VLM backbone + minimal action head; deliberately minimizes architectural complexity to reduce confounders; unified multi-benchmark training
- **Parameters:** Not specified (VLM backbone)
- **Action Head:** Simple MLP head (demonstrates that complex diffusion/flow heads are not always necessary); also tested diffusion-style flow matching variant
- **Training Data:** Unified training across LIBERO, SimplerEnv, RoboTwin, RoboCasa
- **Key Innovation:** Demonstrates that a strong VLM backbone + minimal design is sufficient; questions necessity of complex action heads, robot-specific pretraining, and data engineering; outperforms π₀.5 by 20% on RoboChallenge
- **Key Results:** Outperforms π₀.5 by 20% on public real-world RoboChallenge benchmark; competitive across all benchmarks with simplest architecture
- **Limitations:** May not generalize to all task types; simple MLP head may struggle with highly multi-modal action distributions

---

### Discrete Diffusion VLA (ICLR 2026 submissions)

A cluster of 4 concurrent papers submitted to ICLR 2026 proposing discrete diffusion for VLA action decoding:

- **Discrete Diffusion VLA**: Applies discrete diffusion to OpenVLA for fast action chunk-based generation of discrete action tokens; adaptive decoding for inference; strong on LIBERO + SIMPLER
- **dVLA**: Discrete diffusion VLA with co-generation of future frames and text + actions; ECoT + discrete diffusion; good LIBERO + real-world results
- **DIVA**: Discrete diffusion VLA focusing on token substitution during inference for better performance
- **Unified Diffusion VLA**: Generates future frames and discrete actions together with block-wise causal masking; results on CALVIN, LIBERO, SIMPLER

**Key Innovation:** Parallel generation of action sequences (vs. autoregressive); combines speed of discrete diffusion with quality of diffusion models; enables fast embodied chain-of-thought

---

### MemoryVLA (2025)
- **Venue:** arXiv Aug 2025
- **Architecture:** VLM backbone with perceptual-cognitive memory module for temporal reasoning across manipulation steps
- **Parameters:** Not specified
- **Action Head:** Not specified
- **Training Data:** Robot manipulation demonstrations
- **Key Innovation:** Perceptual-cognitive memory for VLA — maintains and leverages memory of past observations and actions for better long-horizon manipulation
- **Key Results:** Improved performance on long-horizon manipulation tasks
- **Limitations:** Memory module adds complexity and latency

---

### X-VLA (2025)
- **Venue:** arXiv Oct 2025
- **Architecture:** Soft-prompted transformer for scalable cross-embodiment VLA; soft prompts adapt the same backbone to different robot embodiments
- **Parameters:** Not specified
- **Action Head:** Not specified
- **Training Data:** Cross-embodiment robot datasets
- **Key Innovation:** Soft-prompt-based cross-embodiment adaptation; scalable approach to handling diverse robot morphologies without architecture changes
- **Key Results:** Effective cross-embodiment transfer
- **Limitations:** Soft prompt tuning may not capture all embodiment-specific nuances

---

### XR-1 (2025)
- **Venue:** arXiv 2025
- **Architecture:** Learns unified vision-motion representations for versatile VLA; shared representation space for visual observations and motion commands
- **Parameters:** Not specified
- **Action Head:** Not specified
- **Training Data:** Multi-embodiment robot data
- **Key Innovation:** Unified vision-motion representation learning; bridges the gap between visual perception and action generation through shared latent space
- **Key Results:** Versatile performance across multiple robot platforms
- **Limitations:** Representation alignment remains challenging

---

### VLA-RL (2025)
- **Venue:** arXiv May 2025
- **Architecture:** VLA model with reinforcement learning post-training for mastering and generalizing robotic tasks
- **Parameters:** Not specified
- **Action Head:** Inherits from base VLA
- **Training Data:** Base VLA data + RL fine-tuning on target tasks
- **Key Innovation:** Applies RL to post-train VLA models; demonstrates that RL can significantly improve VLA performance beyond imitation learning
- **Key Results:** Improved mastery and generalization over base VLA
- **Limitations:** RL training is expensive and unstable; reward engineering required

---

### EdgeVLA (2025/2026)
- **Venue:** Workshop/preprint
- **Architecture:** Efficient VLA designed for low-power edge devices; model compression and quantization for embedded deployment
- **Parameters:** Sub-500M (target)
- **Action Head:** Efficient action decoder
- **Training Data:** Standard VLA datasets with distillation
- **Key Innovation:** First VLA designed specifically for edge/embodied deployment on low-power hardware; model compression techniques for VLA
- **Key Results:** Enables VLA inference on edge devices
- **Limitations:** Performance trade-off for efficiency; limited task complexity

---

## Architecture Taxonomy

### Action Head Comparison

| Action Head Type | Description | Pros | Cons | Example Models |
|---|---|---|---|---|
| **Discrete Tokens (AR)** | Actions quantized into discrete bins, generated autoregressively like text | Simple; leverages LLM training | Jerky single-step actions; slow sequential generation | RT-2, OpenVLA, RoboFlamingo, ChatVLA |
| **MLP Head** | Continuous actions predicted by MLP from VLM features | Fast inference; simple architecture | Limited expressiveness for multi-modal actions | StarVLA-α (simple variant) |
| **Diffusion (DDPM)** | Iterative denoising from noise to action trajectory | Handles multi-modal distributions; smooth trajectories | Multiple denoising steps (slow); complex training | Octo, DexVLA, Dita |
| **Flow Matching** | Learned velocity field transforms noise to actions | Smoother than DDPM; fewer steps needed; continuous | Still requires multiple denoising steps | π₀, π₀.₅, FLOWER, Xiaomi-Robotics-0 |
| **Chunked Prediction (ACT)** | Predicts sequence of future actions in one forward pass | Fast inference; smooth trajectories | Fixed horizon; may struggle with long tasks | SmolVLA |
| **Discrete Diffusion** | Diffusion in discrete token space; parallel generation | Fast parallel generation; combines AR and diffusion | New paradigm; limited pretrained backbones | Discrete Diffusion VLA, dVLA, DIVA, Unified Diffusion VLA |
| **Frequency-Domain** | Action as time-series forecasting in frequency domain | Efficient; captures periodic motions | Novel approach; limited validation | VLANeXt |

### Backbone Comparison

| Backbone Type | Description | Example Models |
|---|---|---|
| **Large VLM (7B+)** | Llama-2/3, PaLI-X, PaLM-E based | RT-2 (55B), OpenVLA (7.5B), RoboFlamingo (9B), Xiaomi-Robotics-0 (4.7B) |
| **Medium VLM (1-3B)** | Smaller VLMs, often PaLI-based | π₀ (~3B), π₀.₅ (~3B), FLOWER (950M), VLANeXt (2.5B) |
| **Small VLM (<1B)** | Compact VLMs like SmolVLM | SmolVLA (450M), EdgeVLA (<500M target) |
| **Transformer from Scratch** | No internet-scale pretraining | Octo (93M), CrossFormer |
| **Hybrid** | VLM + separate action expert | DexVLA (VLM + 1B diffusion expert) |

---

## Benchmark Comparison

### LIBERO Benchmark (Table-top Manipulation)

| Model | LIBERO-Spatial | LIBERO-Goal | LIBERO-Object | LIBERO-Long | Average |
|---|---|---|---|---|---|
| OpenVLA (fine-tuned) | ~95% | ~95% | ~95% | ~90% | ~93% |
| Xiaomi-Robotics-0 | — | — | — | — | 98.7% |
| VLANeXt (2.5B) | — | — | — | — | SOTA |
| SmolVLA | ~88% | ~88% | ~88% | ~82% | ~87% |
| StarVLA-α | — | — | — | — | Competitive |

*Note: LIBERO is approaching saturation — >95% is expected for Spatial/Goal/Object; 90-95% for Long. A properly tuned Diffusion Policy can achieve competitive results without VLA pretraining.*

### CALVIN Benchmark (Long-horizon Language-conditioned)

| Model | CALVIN ABC | CALVIN ABCD | CALVIN D |
|---|---|---|---|
| FLOWER | **4.53** (SoTA) | — | — |
| π₀ | — | — | — |
| OpenVLA | — | — | — |

*CALVIN ABC >4.0 is standard; >4.5 is SOTA regime. D version: 3.75 standard, >4.0 very good. ABCD: >4.5 relevant.*

### SIMPLER Benchmark

| Model | Google Robot | Bridge V2 |
|---|---|---|
| OpenVLA | ~70-80% | Variable (40-99%) |
| RT-2-X | ~70-80% | — |
| Xiaomi-Robotics-0 | SOTA | — |

*SIMPLER is hard to interpret across setups; success spans 40-99% on Bridge.*

### Real-World RoboChallenge

| Model | Performance |
|---|---|
| StarVLA-α | Outperforms π₀.5 by 20% |
| π₀.₅ | Baseline |

---

## Limitations & Open Problems

### 1. Benchmark Saturation and Evaluation Gaps
- **LIBERO is essentially solved** (>95% for most suites) — no longer discriminative for VLA comparison
- **Sim-only results unreliable** — 7B+ models can overfit simulation benchmarks while failing in the real world
- **Lack of standardized evaluation** — different works use different task definitions, success criteria, and data splits
- **Missing reasoning benchmarks** — existing benchmarks don't test compositional generalization or long-horizon reasoning

### 2. Inference Latency and Real-Time Deployment
- **7B+ models too slow** for reactive tasks (~300ms per action step on consumer GPUs)
- **Multi-step denoising** (diffusion/flow matching) adds 10-20 forward passes per action
- **Action chunk continuity** — chaining consecutive action predictions without jerky transitions remains challenging
- **Consumer GPU deployment** — only SmolVLA (450M) and similar small models achieve real-time on consumer hardware

### 3. Generalization Failures
- **Zero-shot generalization limited** — most VLAs still struggle with truly novel environments, objects, and instructions
- **Semantic gap** — models fail on internet concepts not in robot training data (e.g., "move to Taylor Swift")
- **Sim-to-real transfer** — simulation performance doesn't reliably predict real-world success
- **Cross-embodiment transfer** — aligning observation/action spaces across different robots remains difficult

### 4. Data Infrastructure Challenges
- **Fidelity-cost trade-off** — real-world demonstrations are expensive; synthetic data lacks realism
- **Dataset heterogeneity** — incompatible interfaces, action spaces, and normalization across datasets
- **Scalable data generation** — simulation-based and video-reconstruction engines struggle with physical grounding
- **Representation alignment** — aligning representations across embodiments and modalities is unsolved

### 5. Architecture Design Open Questions
- **Action head choice** — no clear winner among discrete tokens, diffusion, flow matching, MLP, chunked prediction
- **Video inputs** — VLANeXt found video inputs fail to help action learning despite VLM video pretraining
- **Scaling laws** — unclear if bigger models always help; 2.5B VLANeXt outperforms 7B OpenVLA
- **Preserving VL capabilities** — robot training often degrades the VLM's original vision-language understanding

### 6. The Hidden Gap Between Frontier and Academic Labs
- Frontier labs (Google, Physical Intelligence, Figure AI) have access to proprietary datasets orders of magnitude larger than academic benchmarks
- Academic models primarily evaluated on LIBERO/SIMPLER/CALVIN which don't reflect real-world complexity
- Real-world results are rare in academic papers but critical for trust
- The gap is "invisible" from reading papers alone — simulation leaderboards hide it

---

## ICLR 2026 VLA Trends Summary

Based on analysis of 164 VLA submissions at ICLR 2026 (18× increase from ICLR 2025's 9 submissions). Source: Moritz Reuss blog post (Oct 2025), which searched OpenReview for the “Vision-Language-Action” keyword. Note: this is a keyword-based count from a single researcher's search, not an official ICLR statistic — the actual number of VLA-related submissions may differ slightly depending on keyword definitions.

1. **Discrete Diffusion VLAs** — fastest-growing trend; parallel action generation
2. **Reasoning VLAs / Embodied Chain-of-Thought** — reasoning before acting
3. **New Discrete Tokenizers** — better action discretization methods
4. **Efficient VLAs** — sub-1B models achieving competitive performance
5. **RL for VLAs** — reinforcement learning post-training for VLA improvement
6. **VLA + Video Prediction** — generating future frames alongside actions
7. **Evaluation and Benchmarking** — new benchmarks and evaluation protocols
8. **Cross-Action-Space Learning** — handling diverse action representations

---

## References

1. Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale," RSS 2023
2. Zitkovich et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv 2023
3. Octo Model Team et al., "Octo: An Open-Source Generalist Robot Policy," arXiv 2024
4. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv 2024
5. Black et al., "π₀: A Vision-Language-Action Flow Model for General Robot Control," arXiv 2024
6. Intelligence et al., "π₀.₅: a VLA with Open-World Generalization," arXiv 2025
7. Wu et al., "VLANeXt: Recipes for Building Strong VLA Models," arXiv 2026
8. Wen et al., "DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control," CoRL 2025
9. NVIDIA, "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models," arXiv 2025
10. HuggingFace, "SmolVLA: Efficient Vision Language Action Model," 2025
11. Xiaomi Robotics, "Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution," arXiv 2026
12. Hou et al., "Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy," ICCV 2025
13. Reuss et al., "FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies," CoRL 2025
14. Doshi et al., "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation," arXiv 2024
15. Zhou et al., "ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model," EMNLP 2025
16. Zhou et al., "ChatVLA-2: Vision-Language-Action Model with Open-World Reasoning," NeurIPS 2025
17. Ye et al., "StarVLA-α: Reducing Complexity in Vision-Language-Action Systems," arXiv 2026
18. Ma et al., "A Survey on Vision-Language-Action Models for Embodied AI," IEEE TNNLS 2026
19. Reuss, "State of Vision-Language-Action (VLA) Research at ICLR 2026," blog post 2025
20. Wang et al., "Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines," arXiv 2026
21. RoboFlamingo, Shanghai AI Lab, ICRA 2024
22. Figure AI, "Helix," 2025
23. Various ICLR 2026 Discrete Diffusion VLA submissions
24. MemoryVLA, arXiv 2025
25. X-VLA, arXiv 2025
26. XR-1, arXiv 2025
27. VLA-RL, arXiv 2025
28. EdgeVLA, Workshop/preprint 2025-2026
29. Cao et al., "Mamba Policy: Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models," ICRA 2025 (arXiv 2409.07163)

