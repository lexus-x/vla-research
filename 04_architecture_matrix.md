# VLA Architecture Decomposition & Comparison Matrix

> Generated: 2026-07-10 | Sources: arXiv surveys, NVIDIA blog, robotics center comparisons, individual paper deep-dives

---

## 1. Architecture Component Taxonomy

### Visual Encoder Options
| Encoder | Params | Type | Used By |
|---------|--------|------|---------|
| **SigLIP** | 400M | ViT-based CLIP variant | OpenVLA, SmolVLA, OpenPI |
| **PaLI/PALI-X** | Varies (est. 400M-2B enc) | Encoder-decoder VLM | RT-2, π₀ |
| **Eagle** | ~200M | NVIDIA custom ViT | GR00T N1 |
| **Custom ViT** | Small | Task-specific ViT | Octo |
| **DINOv2 + SigLIP** | ~600M (fused) | Dual-encoder fusion | Prismatic (OpenVLA base), MiniVLA |
| **InternViT** | 300M-6B | InternVL family | Some Chinese VLAs |

### Language Encoder Options
| LLM Backbone | Params | Used By |
|-------------|--------|---------|
| **LLaMA 2** | 7B | OpenVLA |
| **Qwen 2.5** | 0.5B | MiniVLA |
| **SmolLM2** | 135M | SmolVLA |
| **PaLI decoder** | ~3B total | π₀, RT-2 |
| **Custom Transformer** | 93M total | Octo |
| **LLaMA-based** | ~3B | OpenPI |

### Fusion Mechanisms
| Mechanism | Description | Used By |
|-----------|-------------|---------|
| **Cross-attention** | Visual tokens attend to language, shared transformer | Octo, CogACT |
| **Token interleaving** | Image/text tokens in same sequence | OpenVLA, RT-2, SmolVLA |
| **MoE (Mixture-of-Experts)** | Separate experts per modality, shared attention | π₀ (action expert) |
| **Dual-system** | Slow VLA + fast policy | GR00T N1 |
| **Fused ViT** | Multi-encoder features merged before LLM | Prismatic/OpenVLA |

### Action Tokenizer / Representation
| Method | Type | Details | Used By |
|--------|------|---------|---------|
| **Discrete binning** | Quantization | 256 bins per dim, 7 dims → 7 tokens | OpenVLA, RT-2 |
| **Flow matching** | Continuous | Straight-line optimal transport, 4-8 denoising steps | π₀, OpenPI, π₀.5 |
| **Diffusion (DDPM)** | Continuous | 16-32 denoising steps | Octo, SmolVLA, Dita |
| **VQ-VAE / RVQ** | Learned codebook | Residual VQ, codebook indices as tokens | MiniVLA (VQ chunking) |
| **FAST** | Learned discrete | Frequency-based action tokenization | OpenVLA-FAST |
| **BEAST** | Learned discrete | Binary-encoded action tokens | Research models |
| **MLP regression** | Direct | Single forward pass, MSE loss | BAKU, simple baselines |

### Action Head Architectures
| Head Type | Architecture | Action Chunk Size | Used By |
|-----------|-------------|-------------------|---------|
| **Next-token prediction** | LLM vocabulary extension | 1 step (7 tokens) | OpenVLA, RT-2 |
| **Small MLP diffusion** | 8-layer MLP conditioned on embedding | 16 steps | Octo |
| **Flow matching MLP** | MLP with flow objective | 50 steps | π₀ |
| **Diffusion Transformer (DiT)** | Full transformer denoiser | 16-64 steps | Dita, CogACT |
| **Hybrid AR+Diffusion** | Dual heads in single LLM | Variable | HybridVLA |
| **Dual head** | Diffusion + MLP fast path | Variable | GR00T N1 |

### Temporal Modeling
| Method | Description | Used By |
|--------|-------------|---------|
| **Action chunking** | Predict H future steps, execute open-loop | π₀ (50 steps), Octo (16), Diffusion Policy |
| **Autoregressive** | Single-step, re-query each timestep | OpenVLA, RT-2 |
| **History frames** | Stack N past observations | MiniVLA (2 frames), Octo (2 frames) |
| **Recurrent hidden state** | LSTM/GRU across timesteps | Early RT models |
| **Real-time chunking** | Overlapping chunks with replanning | π₀-FAST |

---

## 2. Full Architecture Comparison Matrix

### Core VLA Models

| Model | Year | Total Params | Visual Encoder | Lang Encoder | Fusion | Action Tokenizer | Action Head | Temporal | Training Objective |
|-------|------|-------------|----------------|-------------|--------|-----------------|-------------|----------|-------------------|
| **RT-2** | 2023 | 55B | PaLI-X enc | PaLI-X dec | Token interleave | Discrete (256 bins) | Next-token (LLM) | Single-step AR | NLL on action tokens |
| **Octo** | 2024 | 93M | Custom ViT | Shared transformer | Cross-attn readout | Continuous | Small MLP diffusion (16-step chunk) | Chunk (16 steps) | Diffusion denoising |
| **OpenVLA** | 2024 | 7B | SigLIP 400M + DINOv2 | LLaMA 2 7B | Fused ViT → LLM | Discrete (256 bins) | Next-token (LLM) | Single-step AR | NLL on action tokens |
| **π₀** | 2024 | ~3B+ | PaLI-based enc | PaLI dec | MoE action expert | Continuous | Flow matching MLP (50-step chunk) | Chunk (50 steps) | Flow matching |
| **CogACT** | 2024 | ~7B | VLM encoder | VLM decoder | Cross-attn + separate module | Continuous | Diffusion transformer | Chunk (16-64 steps) | Diffusion denoising |
| **SmolVLA** | 2025 | 500M | SigLIP 400M | SmolLM2 135M | Token interleave | Continuous | Diffusion head | Chunk | Diffusion denoising |
| **GR00T N1** | 2025 | ~2B | Eagle | Custom | Dual system | Continuous | Dual (diffusion + MLP fast) | Chunk + fast policy | Diffusion + MSE |
| **Dita** | 2025 | ~3B | VLM encoder | VLM decoder | In-context conditioning | Continuous | Large DiT denoiser | Chunk (64 steps) | Diffusion denoising |
| **HybridVLA** | 2025 | ~7B | VLM encoder | VLM decoder | Unified LLM | Both discrete + continuous | Dual AR + diffusion in single LLM | Both AR + chunk | NLL + diffusion jointly |
| **MiniVLA** | 2024 | 1B | SigLIP + DINOv2 | Qwen 2.5 0.5B | Fused ViT → LLM | VQ-RVQ codebook | Next-token (LLM + VQ codes) | Chunk (8 steps via VQ) | NLL on VQ tokens |
| **OpenPI** | 2025 | ~3B | SigLIP 400M | LLaMA-based | Token interleave + action expert | Continuous | Flow matching MLP | Chunk (50 steps) | Flow matching |
| **π₀.5** | 2025 | ~3B+ | PaLI-based | PaLI dec | MoE + high-level policy | Continuous | Flow matching | Chunk (50 steps) | Flow matching + HL policy |
| **OneVLA** | 2026 | ~7B | VLM encoder | VLM decoder | Unified action head | Continuous | Unified nav+manip head | Chunk | Multi-stage progressive |
| **OpenVLA-FAST** | 2025 | 7B | SigLIP + DINOv2 | LLaMA 2 7B | Fused ViT → LLM | FAST (learned discrete) | Next-token (LLM + FAST codes) | Chunk via FAST | NLL on FAST tokens |

### WAM (World-Action Models) - Emerging Paradigm

| Model | Year | Backbone | Action Integration | Paradigm |
|-------|------|----------|-------------------|----------|
| **UniPi** | 2023 | Video diffusion | Inverse dynamics | Video prediction → action |
| **VPP** | 2024 | Video predictor | Representation conditioning | Predictive visual repr |
| **DreamZ** | 2025 | Video backbone (WAN) | Action tokens | Video + action joint |
| **LAPA** | 2024 | Video model | Latent actions | Unsupervised action from video |

---

## 3. Component Interaction Analysis

### 3.1 Visual Encoder Impact

| Configuration | Strengths | Weaknesses |
|--------------|-----------|------------|
| **SigLIP (400M)** | Good semantic grounding, well-understood | Fixed resolution, no spatial inductive bias |
| **DINOv2 + SigLIP fused** | Best of both: semantic + spatial features | 2x encoder compute, larger model |
| **Custom small ViT** | Fast inference, small footprint | Limited pre-trained knowledge |
| **PaLI encoder** | Tightly coupled with language decoder | Locked to specific LLM family |

**Key finding:** The Prismatic dual-encoder (DINOv2 + SigLIP) used in OpenVLA provides strong visual features. MiniVLA retains this encoder while shrinking the LLM, showing the visual encoder is not the bottleneck.

### 3.2 Action Head Trade-offs

| Head Type | Precision | Speed | Multimodal | Chunk Smoothness |
|-----------|-----------|-------|------------|-----------------|
| **Discrete bins (256)** | Medium (quantization error) | Fast (1 pass) | No (unimodal) | Poor (independent dims) |
| **MLP diffusion** | High | Slow (16-32 steps) | Yes | Good |
| **Flow matching** | High | Medium (4-8 steps) | Yes | Excellent |
| **Large DiT** | Highest | Slowest (full transformer) | Yes | Excellent |
| **VQ-RVQ** | Medium-High | Fast (codebook lookup) | Limited | Good (learned chunks) |
| **Hybrid AR+Diff** | Highest | Medium | Yes (both) | Best of both |

**Key finding:** Flow matching (π₀) achieves diffusion-quality outputs with 4-8 steps vs 16-32 for DDPM, making it the current sweet spot for quality vs speed.

### 3.3 Fusion Mechanism Impact

| Mechanism | Cross-modal understanding | Compute overhead | Scalability |
|-----------|--------------------------|------------------|-------------|
| **Token interleaving** | Good (same attention) | Low | Excellent (standard LLM) |
| **Cross-attention readout** | Better (dedicated attention) | Medium | Good |
| **MoE action expert** | Best (specialized + shared) | Medium-High | Excellent |
| **Dual-system** | Separate concerns | High (2 models) | Good |

---

## 4. Inference Pipeline Latency Breakdown

Based on profiling studies (Zhou et al. 2026, NVIDIA blog):

### Two-Phase Inference Pattern
```
Phase 1: VLM Backbone (compute-bound)
  - Visual encoding: 5-15ms (SigLIP)
  - LLM forward pass: 50-200ms (7B model)
  - Total Phase 1: ~60-200ms

Phase 2: Action Expert (memory-bound)
  - Diffusion/flow denoising: 20-100ms (4-32 steps)
  - Action decoding: 1-5ms
  - Total Phase 2: ~20-100ms
```

### End-to-End Latency by Model

| Model | Hardware | Latency | Frequency | Action Chunk |
|-------|----------|---------|-----------|-------------|
| **OpenVLA (7B)** | A100 | ~200ms | 5 Hz | 1 step (re-query) |
| **OpenVLA (7B)** | Jetson Orin | ~500ms | 2 Hz | 1 step |
| **Octo (93M)** | A100 | ~20ms | 50 Hz | 16 steps |
| **π₀ (~3B)** | A100 | ~30-50ms | 20-50 Hz | 50 steps |
| **SmolVLA (500M)** | Jetson NX | ~50ms | 20 Hz | chunk |
| **MiniVLA (1B)** | L40s | ~80ms | 12.5 Hz | 8 steps (VQ) |
| **GR00T N1** | Jetson Thor | ~5ms (fast path) | 200 Hz | fast policy |

**Key bottleneck:** For 7B-class models, the LLM backbone dominates latency. Action chunking amortizes this cost by executing multiple steps per inference.

---

## 5. Training Objectives Comparison

| Objective | Description | Used By | Properties |
|-----------|-------------|---------|------------|
| **NLL on discrete tokens** | Cross-entropy loss on binned actions | OpenVLA, RT-2 | Simple, leverages LLM pretraining |
| **Diffusion denoising** | MSE on noise prediction | Octo, SmolVLA, CogACT, Dita | Handles multimodal distributions |
| **Flow matching** | Optimal transport velocity field | π₀, OpenPI | Faster convergence, fewer steps |
| **VQ reconstruction** | Codebook + reconstruction loss | MiniVLA | Discrete but learned representation |
| **Hybrid NLL + Diffusion** | Joint loss on both heads | HybridVLA | Best of both paradigms |
| **Multi-stage progressive** | Manipulation → Navigation → CoT | OneVLA | Cross-task transfer |

---

## 6. Unexplored & Underexplored Combinations

### 6.1 Visual Encoder × Action Head Gaps

| Visual Encoder | Action Head | Status | Potential |
|---------------|-------------|--------|-----------|
| **DINOv2 standalone** | Flow matching | ❌ Unexplored | DINOv2's spatial features may better ground fine motor control |
| **SigLIP** | VQ-RVQ learned codebook | ⚠️ Only MiniVLA | Could combine with larger LLMs |
| **Eagle (NVIDIA)** | Flow matching | ❌ Unexplored | GR00T uses diffusion; flow matching could be faster |
| **InternViT (6B)** | Any action head | ❌ Unexplored | Largest open vision encoder, no VLA yet |
| **Video encoder (ViViT)** | Flow matching | ⚠️ WAM only | Temporal visual features + continuous actions |
| **Multi-scale FPN** | Diffusion | ❌ Unexplored | Multi-resolution features for different action granularities |

### 6.2 Language Encoder × Fusion Gaps

| Language Model | Fusion | Status | Potential |
|---------------|--------|--------|-----------|
| **Qwen 2.5 (72B)** | Any | ❌ Unexplored | Strongest open LLM, no VLA built on it |
| **Mamba/SSM** | Any | ❌ Unexplored | Linear attention for lower latency |
| **Multimodal LLM (native)** | Direct action | ⚠️ Early | Models like Qwen-VL that natively handle images |
| **Code LLM** | Structured action output | ❌ Unexplored | Code generation as action specification |

### 6.3 Action Head × Temporal Model Gaps

| Action Head | Temporal | Status | Potential |
|------------|----------|--------|-----------|
| **Flow matching** | Recurrent (hidden state) | ❌ Unexplored | Flow matching with temporal memory |
| **VQ-RVQ** | Flow matching refinement | ❌ Unexplored | Coarse VQ + fine flow refinement |
| **Hybrid AR+Diff** | Real-time chunking | ⚠️ π₀-FAST only | HybridVLA hasn't tried RT chunking |
| **DiT** | History-conditioned | ⚠️ Dita only | Large denoiser with visual history |

### 6.4 Paradigm-Level Gaps

| Combination | Status | Potential Impact |
|------------|--------|-----------------|
| **WAM backbone + VLA action head** | ⚠️ Early (DreamZ) | Video prior + precise actions |
| **DINOv2 + Flow matching + SSM language** | ❌ Completely unexplored | Fastest possible pipeline |
| **Multi-resolution action (coarse-to-fine)** | ❌ Unexplored | Hierarchical: plan → refine |
| **Action head that predicts corrections** | ❌ Unexplored | Residual policy on top of base |
| **Cross-embodiment action alignment** | ⚠️ Limited | Shared latent action space across robots |
| **Mixture-of-Experts per action dimension** | ❌ Unexplored | Different experts for position vs gripper |

---

## 7. Key Insights & Design Principles

### What Works (Evidence-Based)

1. **Flow matching > Diffusion for speed**: π₀ achieves comparable quality with 4-8 steps vs 16-32 (2-4x faster inference)
2. **Action chunking is critical**: All top models predict 16-50 steps at once; single-step prediction underperforms
3. **Dual-encoder visual features help**: DINOv2 + SigLIP (Prismatic) outperforms single encoders
4. **Larger LLM backbone ≠ always better**: MiniVLA (0.5B LLM) matches OpenVLA (7B) with better action representations
5. **Discrete tokenization is a bottleneck**: Continuous action heads (diffusion/flow) outperform 256-bin discretization by 15-35%
6. **MoE action experts scale well**: π₀'s separate action expert allows independent scaling

### What Doesn't Work (Known Failures)

1. **Single-step MLP regression**: Too simple for multimodal action distributions
2. **Very small transformers (93M)**: Octo struggles with spatial reasoning and novel objects
3. **Pure autoregressive for high-frequency control**: Too slow for 50Hz; need action chunking
4. **Fixed action binning**: 256 bins insufficient for dexterous tasks requiring sub-mm precision

### The Emerging Hybrid Recipe (2025-2026)

The field is converging on:
```
Visual: DINOv2 + SigLIP dual encoder (or Eagle)
Language: 1-7B LLM (balanced for speed)
Fusion: Token interleaving or MoE
Action: Flow matching or large DiT (continuous, chunked)
Temporal: 16-50 step action chunks
Training: Pre-train on OXE → Fine-tune on target
```

---

## 8. Research Priorities (Ranked by Impact × Feasibility)

| Priority | Research Direction | Expected Impact |
|----------|-------------------|-----------------|
| 🔴 **High** | Flow matching + SSM backbone for real-time VLA | 5-10x latency reduction |
| 🔴 **High** | Multi-resolution action prediction (coarse plan + fine motor) | Better long-horizon tasks |
| 🟡 **Medium** | DINOv2 spatial features for precise manipulation | Better spatial grounding |
| 🟡 **Medium** | Cross-embodiment latent action alignment | Universal action space |
| 🟡 **Medium** | VQ-RVQ + flow matching hybrid | Fast coarse + precise fine |
| 🟢 **Low** | Code LLM for structured action generation | Novel action specification |
| 🟢 **Low** | MoE per action dimension | Specialized position/gripper |

---

## 9. Quick Reference: Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Research baseline** | OpenVLA | Open, well-documented, 7B |
| **Fast iteration / limited compute** | Octo or MiniVLA | Small, fast fine-tuning |
| **State-of-the-art performance** | π₀ / π₀.5 | Flow matching, best results |
| **Consumer hardware** | SmolVLA | 500M, runs on Jetson NX |
| **Humanoid / NVIDIA ecosystem** | GR00T N1 | Dual-system, sim-first |
| **Best of both paradigms** | HybridVLA | AR reasoning + diffusion precision |
| **Navigation + manipulation** | OneVLA | Unified head, cross-task transfer |
| **Lowest latency** | Octo (93M) | 50Hz on A100, 20ms inference |

---

## References

- Kim et al. (2024) "OpenVLA: An Open-Source Vision-Language-Action Model" arXiv:2406.09246
- Black et al. (2024) "π₀: A Vision-Language-Action Flow Model" arXiv:2410.24164
- Octo Model Team (2024) "Octo: An Open-Source Generalist Robot Policy" arXiv:2405.12213
- Bjorck et al. (2025) "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots" arXiv:2503.14734
- Liu et al. (2025) "HybridVLA: Collaborative Diffusion and Autoregression" arXiv:2503.10631
- Li et al. (2024) "CogACT: A Foundational VLA Model" arXiv:2411.19650
- Hou et al. (2025) "Diffusion Transformer Policy" arXiv:2410.15959
- Belkhale & Sadigh (2024) "MiniVLA" Stanford SAIL Blog
- Zhou et al. (2026) "Characterizing VLAs across XPUs" arXiv:2604.24447
- Reuss (2026) "Pretrained to Imagine, Fine-Tuned to Act" NVIDIA Blog
- Sapkota et al. (2025) "VLA Models: Concepts, Progress" arXiv:2505.04769
- Zhang et al. (2026) "OneVLA: A Unified Framework" arXiv:2606.01241
- Pertsch et al. (2025) "FAST: Efficient Action Tokenization" arXiv:2501.09747
- Black et al. (2025) "π₀.5" arXiv:2504.16054
