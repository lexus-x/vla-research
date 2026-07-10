# VLA Research Novelty Database

> Last updated: 2026-07-10
> Purpose: Track existing VLA work to identify novelty gaps

## Key Surveys & Overviews

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| State of VLA Research at ICLR 2026 | 2025 | Blog/ICLR | 164 VLA submissions at ICLR 2026 (18x YoY growth); taxonomy of VLA trends | Survey | N/A | Identified trends: discrete diffusion VLAs, embodied CoT, efficient VLAs, new tokenizers |
| An Anatomy of VLA Models: Modules to Milestones | 2025 | arXiv | Comprehensive anatomy of VLA architectures, from modules to milestones | Survey | N/A | Systematic taxonomy of VLA components |
| World Model for Robot Learning: Comprehensive Survey | 2026 | arXiv | Surveys world models applied to robot learning | Survey | N/A | Covers dynamics models for manipulation |

## Core VLA Models

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| RT-2 (Robotic Transformer 2) | 2023 | CoRL | Fine-tune VLM (PaLI-X/PaLM-E) to output robot actions as text tokens | Architecture | 55B | Strong web-scale generalization to robotic tasks |
| Octo | 2024 | RSS | Transformer-based robot policy trained on Open X-Embodiment dataset | Training/Data | 93M | Cross-embodiment transfer on 25+ datasets |
| OpenVLA | 2024 | NeurIPS | Open-source VLA based on Prismatic VLM, fine-tuned for robot action | Architecture/Training | 7B | SOTA on SIMPLER/real-world tasks; open-source |
| π₀ (Pi-Zero) | 2024 | Physical Intelligence | Flow matching diffusion head on VLM backbone for continuous actions | Architecture | 3B | Strong real-world manipulation performance |
| π₀.₆ | 2026 | RSS | VLA that learns from experience (online RL on top of VLA) | Training | ~3B | Improves π₀ with experience-based learning |
| HPT (Heterogeneous Pretrained Transformers) | 2024 | NeurIPS | Stem-trunk-stem architecture for cross-embodiment pretraining | Architecture | 300M+ | Trained on 52 datasets, transfers across embodiments |
| SpatialVLA | 2025 | arXiv | Spatially-aware VLA with 3D spatial reasoning | Architecture | 7B | Better spatial understanding for manipulation |
| DexVLA | 2025 | arXiv | VLA for dexterous manipulation with diffusion action head | Architecture | 7B | Dexterous bimanual manipulation |
| TraceVLA | 2025 | arXiv | Visual trace conditioning for VLA | Architecture | 7B | Improved spatial grounding via visual traces |
| RoboVLMs | 2025 | arXiv | Unified framework for adapting VLMs to VLAs | Architecture | Various | Benchmark of VLM-to-VLA adaptation strategies |
| VideoVLA | 2025 | NeurIPS | Transforms large video generation models into robotic VLA manipulators | Architecture | Large | Leverages video generation for generalizable robot manipulation |
| ET-VLA | 2025 | arXiv | Embodiment transfer learning with Synthetic Continued Pretraining (SCP) + Embodied Graph-of-Thought for multi-robot | Training | ~7B | Outperforms OpenVLA by 53.2% on 6 real-world bimanual tasks |
| ForceVLA | 2025 | NeurIPS | Force-aware MoE fusion module treating force sensing as first-class modality in VLA | Architecture | ~7B | Contact-rich manipulation with force/tactile sensing integration |

## Discrete Diffusion VLAs (ICLR 2026 trend)

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| Discrete Diffusion VLA | 2025 | ICLR 2026 sub | Combines discrete tokenization with diffusion for action generation | Architecture | ~3B | Bridges discrete language and continuous action spaces |

## Embodied Chain-of-Thought / Reasoning VLAs

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| Embodied CoT (ECoT) | 2024 | CoRL 2025 | Trains VLA to autoregressively generate textual embodied reasoning (grounded in sensory observations) before action | Inference/Training | 7B | +28% absolute success rate on OpenVLA; outperforms RT-2-X on generalization tasks |

## Efficient / Compact VLAs

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| SmolVLA | 2025 | arXiv | 450M-param VLA (SigLIP+SmolLM2+diffusion); async inference; community-driven | Architecture/Inference | 450M | Performance comparable to 10x larger VLAs; runs on consumer GPUs/CPUs |
| TinyVLA | 2025 | RA-L | Compact VLA family using robust multimodal backbone + diffusion policy decoder; no pretraining needed | Architecture/Training | ~300M | Faster inference and better data efficiency than OpenVLA; strong generalization |
| NanoVLA | 2025 | arXiv | Nano-scale VLA with vision-language decoupling, long-short action chunking, and dynamic routing based on task complexity | Architecture/Inference | <500M | Optimized for Jetson Orin Nano; adaptive backbone selection |
| EfficientVLA | 2025 | arXiv | Token compression and pruning for VLA efficiency | Inference | Various | Reduced compute with minimal performance loss |
| BitVLA | 2025 | arXiv | Fully native 1-bit VLA (ternary {-1,0,1}) built on BitNet b1.58; Quantize-then-Distill for vision encoder | Architecture/Inference | 2B (1-bit) | Matches OpenVLA-OFT with 11x memory reduction and 4.4x latency reduction |
| LiteVLA-Edge | 2026 | arXiv | 4-bit quantized VLA pipeline for Jetson Orin embedded hardware with llama.cpp runtime | Inference | ~7B (4-bit) | 150.5ms latency (6.6Hz) on embedded hardware, fully offline |
| RLRC | 2025 | arXiv | RL-based recovery for compressed VLAs (quantization/pruning recovery via reinforcement learning) | Training | Various | Recovers performance lost during compression |

## Cross-Embodiment / Universal VLAs

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| HPT | 2024 | NeurIPS | Stem-trunk-stem for heterogeneous embodiments | Architecture | 300M+ | 52-dataset cross-embodiment training |
| Octo | 2024 | RSS | Open X-Embodiment cross-embodiment policy | Training/Data | 93M | Generalist policy across robot morphologies |
| UniPi | 2024 | ICLR | Universal policy via video generation | Architecture | 1B+ | Text-conditioned video as unified action space |
| UniAct | 2025 | arXiv | Universal Action Space framework capturing generic atomic behaviors across diverse robots | Architecture/Training | Various | Cross-embodiment foundation model via learned universal actions |
| X-VLA | 2025 | OpenReview | Soft-prompted transformer for scalable cross-embodiment VLA | Architecture | ~7B | Soft prompts per data source for cross-robot transfer |
| HEX | 2026 | arXiv | Humanoid-aligned experts for cross-embodiment whole-body manipulation with coordinated body control | Architecture | ~7B | Whole-body humanoid control with MoE experts per body part |
| WholebodyVLA | 2026 | ICLR | Unified latent action space for humanoid robots | Architecture | ~7B | ICLR 2026; humanoid-specific VLA |

## Action Head Designs

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| π₀ | 2024 | arXiv | Flow matching diffusion as action head on VLM | Architecture | 3B | Smooth continuous action generation |
| OpenVLA | 2024 | NeurIPS | Discrete action tokenization (bins) | Architecture | 7B | Simple but effective discrete action output |
| SpatialVLA | 2025 | arXiv | 3D spatial action prediction head | Architecture | 7B | Better spatial grounding |
| StarVLA-α | 2026 | arXiv | Simple baseline for systematic VLA design choice evaluation; minimal architecture complexity | Architecture | ~3B | Controlled ablation of action modeling, pretraining, and interface engineering |
| X-DiffVLA | 2026 | arXiv | Diffusion-based cross-embodied action head with embodiment-specific modulation | Architecture | ~7B | Cross-embodiment transfer with shared VLM backbone + heterogeneous end-effectors |
| VOTE | 2025 | arXiv | Efficient VLA architecture that generates fewer action tokens | Architecture/Inference | Various | Reduced action token generation for faster inference |

## World Model + VLA / Planning

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| PVI (Plug-in Visual Injection) | 2026 | arXiv | Lightweight encoder-agnostic module injecting temporal/spatial visual features into VLA action expert | Architecture | Small (plug-in) | Improves VLA with fine-grained geometric and temporal cues without modifying backbone |
| SPARKS | 2026 | arXiv/LinkedIn | VLA-agnostic memory module with active temporal sampling for robot policies | Architecture | Small (plug-in) | Non-uniform temporal sampling improves memory-based VLA policies |
| EgoRoC | 2026 | CVPR | Task-agnostic visual alignment decoupling how robots see from how they act | Architecture | Plug-in | Decouples perception from action in VLA systems |
| Long-VLA | 2025 | OpenReview | Agnostic module for unleashing long-horizon capability in VLAs | Architecture | Plug-in | Seamless integration into existing VLAs for long-horizon tasks |
| LiLo-VLA | 2026 | arXiv | Modular framework: Reaching Module + object-centric Interaction Module for compositional long-horizon manipulation | Architecture | Modular | Zero-shot generalization to novel long-horizon tasks; robust failure recovery |
| ForceVLA | 2025 | NeurIPS | Force-aware MoE fusion module treating force sensing as first-class modality in VLA | Architecture | ~7B | Contact-rich manipulation with force/tactile sensing integration |
| Action-Specialized MoE (AS-MoE) | 2025 | arXiv | Converts pretrained dense VLA to sparse MoE; action-specialized experts with load balancing | Architecture | ~7B (sparse) | Efficient scaling of VLA via MoE without retraining from scratch |
| MoE-ACT | 2026 | arXiv | Sparse MoE in ACT Transformer encoder for multi-task bimanual manipulation; FiLM modulation + multi-scale cross-attention | Architecture | ~100M | Lightweight multi-task framework; decouples action distributions via expert activation |
| GeRM | 2024 | arXiv | MoE-based VLA for quadruped robots using offline RL + multi-modal transformer | Architecture/Training | ~3B | MoE enables faster inference with higher capacity; multi-task quadruped control |
| DiTEA | 2025 | AAAI | MoE architecture in VLA backbone with RL fine-tuning | Architecture/Training | ~7B | AAAI 2025; MoE + RL for large-scale VLA |

## State Space Models / Mamba-based VLAs

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| RoboMamba | 2024 | NeurIPS | End-to-end VLA using Mamba SSM backbone for linear inference complexity; Ego-3D positioning for spatial reasoning | Architecture | ~2B (Mamba) | Efficient VLA with linear-time inference; strong reasoning + manipulation |
| AnoleVLA | 2026 | arXiv | Lightweight VLA with deep SSM (Mamba-style) for mobile manipulation; resource-constrained deployment | Architecture | Small (SSM) | Efficient sequential state modeling for mobile robots; real-world validated |

## Novel/Variation Architectures

| Paper | Year | Venue | Core Idea | Category | Model Size | Key Result |
|-------|------|-------|-----------|----------|------------|------------|
| GR00T N1 | 2025 | NVIDIA | Dual action head (diffusion + MLP) with Eagle vision encoder for humanoid whole-body control | Architecture | ~2B | NVIDIA's humanoid robot foundation model |
| ExpReS-VLA | 2025 | arXiv | Experience replay + retrieval-augmented generation for rapid on-device VLA specialization; 97% storage reduction via frozen vision embeddings | Training/Inference | ~7B | Prevents catastrophic forgetting; on-device adaptation with compressed experience buffer |
| MAP-VLA | 2025 | arXiv | Memory-augmented prompting with demonstration-derived soft prompts retrieved at inference for long-horizon tasks | Inference | ~7B | Enables long-horizon manipulation via learnable memory prompts |
| SuSIE (Subgoal Synthesis via Image Editing) | 2024 | arXiv | World model generates subgoal images for VLA | Inference | 1B+ | Improved long-horizon task success |
| UniSim | 2023 | NeurIPS | Universal simulator via video generation for robot learning | Training | 1B+ | Synthetic data generation for policy training |
| Genie 2 | 2025 | DeepMind | World model for generating interactive environments | Training/Data | Large | Controllable 3D environment generation |
| TriVLA | 2025 | arXiv | Triple-system VLA: VLM (System 2) + video diffusion (System 1) + episodic world model for memory/recall/prediction | Architecture | ~7B | First formalized episodic world model in VLA; general robot control with temporal reasoning |
| Self-Improving Robot Policy with Compositional World Model | 2026 | RSS | Self-improving policy via compositional world model for robot learning | Training | N/A | RSS 2026 accepted; compositional world models for policy improvement |

---

## Summary Statistics

| Category | Count | Key Trends |
|----------|-------|------------|
| Architecture | ~25 papers | MoE, SSM/Mamba, diffusion heads, modular designs, cross-embodiment |
| Training | ~10 papers | Experience replay, synthetic pretraining, RL fine-tuning, distillation |
| Inference | ~10 papers | Quantization (1-bit, 4-bit), RAG, memory-augmented prompting, async inference |
| Data | ~5 papers | Universal action spaces, video generation, world models |
| Survey | ~5 papers | Rapid growth: 164 VLA submissions at ICLR 2026 (18x YoY) |

## Novelty Gap Analysis

### Heavily Explored (crowded)
- **Large VLA models (7B+)**: OpenVLA, π₀, SpatialVLA, DexVLA, etc.
- **Diffusion action heads**: π₀, X-DiffVLA, TinyVLA all use diffusion/flow matching
- **MoE for VLA**: ForceVLA, AS-MoE, MoE-ACT, GeRM, DiTEA (5+ papers)
- **Quantization/compression**: BitVLA, LiteVLA-Edge, RLRC (well-covered)
- **Cross-embodiment**: HPT, Octo, UniAct, X-VLA, HEX, X-DiffVLA (6+ papers)
- **Embodied CoT**: ECoT established; emerging trend at ICLR 2026

### Moderately Explored
- **Small/efficient VLA**: SmolVLA (450M), TinyVLA, NanoVLA (3 papers)
- **VLA + world models**: TriVLA, SuSIE, compositional world models (3-4 papers)
- **RAG for VLA**: ExpReS-VLA, MAP-VLA (2 papers)
- **SSM/Mamba VLA**: RoboMamba, AnoleVLA (2 papers)
- **Plug-in modules**: PVI, SPARKS, EgoRoC, Long-VLA (4 papers)
- **Force/tactile VLA**: ForceVLA (1 paper, NeurIPS 2025)

### Under-Explored (novelty opportunities)
- **VLA + retrieval from external knowledge bases** (only 2 papers, both recent)
- **SSM-based VLA** (only RoboMamba + AnoleVLA; Mamba-2/V2 not yet applied)
- **VLA with explicit memory architectures** (TriVLA is only formal episodic memory)
- **VLA + reinforcement learning from human feedback (RLHF)** (no papers found)
- **VLA for deformable objects** (no specialized work found)
- **VLA with test-time compute scaling** (no papers found; LLM test-time scaling not applied to VLA)
- **VLA + graph neural networks** (no papers found)
- **Multimodal VLA (audio + tactile + force + vision)** (only ForceVLA for force)
- **VLA with causal reasoning** (no papers found)
- **VLA for multi-agent collaboration** (only ET-VLA for bimanual)
- **Hierarchical VLA (high-level planner + low-level executor)** (emerging but sparse)
- **VLA with uncertainty estimation / calibrated confidence** (no papers found)
- **VLA + active perception / gaze control** (no papers found)

### Highest Novelty Potential
1. **SSM-based VLA with Mamba-2/V2** — only 2 papers, both using original Mamba
2. **VLA with test-time compute scaling** — directly transferable from LLM research
3. **VLA + RLHF / preference optimization** — unexplored in VLA domain
4. **Multimodal VLA (audio + tactile + force)** — only ForceVLA touches force
5. **VLA with explicit uncertainty quantification** — critical for safety
6. **VLA + active perception** — no dedicated work found
7. **VLA for deformable/dynamic objects** — largely unexplored specific niche
