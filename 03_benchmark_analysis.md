# VLA Benchmark Analysis

*Generated: 2026-07-10*

---

## 1. LIBERO Benchmark

**Overview:** The most widely adopted VLA evaluation suite. Kitchen tabletop manipulation tasks in simulation (SAPIEN). De facto standard for VLA comparison.

**Task Suites (4):**
- **Spatial** — tasks requiring spatial reasoning (e.g., "put the bowl on the left burner")
- **Object** — object-centric manipulation (e.g., "pick up the ketchup")
- **Goal** — goal-conditioned tasks
- **Long** — long-horizon multi-step tasks

**Difficulty Levels:** 10 tasks per suite, 40 tasks total. Each task has 50 training demonstrations and 20 evaluation episodes.

### SOTA Results (Standard LIBERO)

| Model | Size | Goal | Object | Spatial | Long | Average |
|-------|------|------|--------|---------|------|---------|
| pi0.5 | ~7B | 0.98 | 0.98 | 0.98 | 0.92 | **0.97** |
| GeoVLA | ~7B | 0.97 | 0.99 | 0.98 | 0.97 | **0.98** |
| OpenVLA | ~7B | 0.98 | 0.99 | 0.98 | 0.93 | **0.97** |
| CogACT | ~7B | 0.90 | 0.98 | 0.97 | 0.89 | **0.98** |
| pi0 | ~7B | 0.92 | 0.98 | 0.97 | – | **0.96** |
| UniVLA* | ~8B | 0.94 | 0.99 | 0.95 | 0.94 | **0.96** |
| SmolVLA | ~2B | 0.91 | 0.94 | 0.93 | 0.73 | **0.86** |
| SpatialVLA | ~4B | 0.79 | 0.90 | 0.88 | 0.56 | **0.78** |
| Octo | ~93M | 0.85 | 0.86 | 0.79 | 0.51 | **0.75** |

**Where models fail most:** Long-horizon tasks are consistently the hardest (5-15% lower than other suites). Spatial tasks are second hardest.

### LIBERO-PRO: Critical Flaw Exposed

LIBERO-PRO (2025) systematically evaluates under perturbations across 4 dimensions: **manipulated objects, initial states, task instructions, and environments**.

**Key finding:** Models achieving >90% on standard LIBERO **collapse to 0.0%** under perturbations. Models exhibit rote memorization — they execute identical trajectories even when:
- Target object is replaced with unrelated items
- Object positions change
- Instructions are corrupted with nonsense tokens
- Object is removed entirely

**Implication:** Standard LIBERO scores significantly overestimate VLA capability. Models memorize action sequences rather than understanding tasks.

---

## 2. LIBERO-plus (LIBERO+)

**Overview:** CVPR 2026. Extended LIBERO with controlled perturbations across **7 dimensions** to test robustness. Difficulty levels L1–L5 for fine-grained failure analysis.

**7 Perturbation Dimensions:**
1. Objects Layout — add confounding objects, shift target positions
2. Camera Viewpoints — change viewpoint/pose/FOV
3. Robot Initial States — change manipulator initial pose
4. Language Instructions — rewrite with richer/complex phrasing
5. Light Conditions — vary intensity, direction, color, shadows
6. Background Textures — modify table/scene textures
7. Sensor Noise — photometric distortions (jitter, Gaussian blur)

### Model Robustness Results (% success rate under perturbation)

| Model | Original | Camera | Robot | Language | Light | Background | Noise | Layout |
|-------|----------|--------|-------|----------|-------|------------|-------|--------|
| OpenVLA | 76.5 | **1.1** | **4.1** | 26.8 | 4.4 | 25.3 | 19.3 | 31.6 |
| OpenVLA-OFT | 97.1 | 59.7 | **37.2** | 81.5 | 85.8 | 92.4 | 76.7 | 77.1 |
| π₀ | 94.2 | **15.8** | **6.6** | 61.0 | 79.6 | 78.5 | 79.4 | 70.4 |
| π₀-fast | – | – | – | – | – | – | – | – |
| Nora | – | – | – | – | – | – | – | – |
| WorldVLA | – | – | – | – | – | – | – | – |
| UniVLA | – | – | – | – | – | – | – | – |

### Critical Findings

**Worst perturbation dimensions:**
- **Camera viewpoints** — most devastating; OpenVLA drops 75.4 points, π₀ drops ~78 points
- **Robot initial states** — second worst; OpenVLA-OFT drops 59.9 points
- **Language instructions** — models largely **ignore language**; removing instructions causes minimal further drop

**Surprise finding:** Models are **largely insensitive to language variations**, suggesting they don't truly follow instructions. Further experiments confirm models tend to ignore language instructions completely.

### VLANeXt Results (SOTA on LIBERO-plus, 2026)

VLANeXt (2.5B params) outperforms OpenVLA-OFT (7B) on both LIBERO and LIBERO-plus.

**Key findings from VLANeXt's 500+ ablation experiments:**
1. Separate policy head > reusing text tokens
2. Longer action chunking (8 steps) consistently improves performance
3. **Regression > diffusion > classification** for action learning
4. Video inputs do NOT help action learning even with video-pretrained VLMs
5. Conditioning proprioception in VLM > omitting or injecting directly to policy

### Common Failure Modes

1. **Camera sensitivity** — models overfit to fixed viewpoints; slight shifts cause catastrophic failure
2. **Kinematic reasoning weakness** — can't generalize across robot initial configurations
3. **Language blindness** — models essentially ignore instructions, relying on visual patterns
4. **Object layout brittleness** — confounding objects or position shifts confuse grasping
5. **Long-horizon fragility** — compounding errors in multi-step tasks under distribution shift

---

## 3. MetaWorld Benchmark

**Overview:** 50 manipulation tasks in MuJoCo (sawyer_robot arm). Widely used for single-task and multi-task RL evaluation. Increasingly used for VLA evaluation.

**Task Categories:**
- Reach, Push, Pick-and-Place, Peg-in-Hole
- Drawer open/close, Window open/close
- Button press, Lever pull
- 50 distinct tasks total

### MetaWorld Results (VLA models)

| Model | Size | Avg Success Rate | Notes |
|-------|------|-----------------|-------|
| Evo-1 (CVPR 2026) | ~1B | 85-92% | Lightweight, no robot pretraining |
| ProgVLA (2026) | ~100M | 80-88% | 0.1B params, progress-aware |
| ActionX (2026) | ~3B | +16% over baselines | RL pre-trained action experts |
| OpenVLA | 7B | 75-85% | Standard baseline |
| Octo | 93M | 65-75% | Smaller but less capable |

### MetaWorld Failure Modes
- **Precision tasks** (peg-in-hole): lowest success rates across all models
- **Long-horizon** (multi-step): compounding errors
- **Sparse reward tasks**: models struggle without dense supervision

---

## 4. Inference Latency Comparison

| Model | Size | Latency (ms/step) | Hz | Hardware |
|-------|------|-------------------|-----|----------|
| RT-2 | 55B | ~500ms | 2 | TPUv4 |
| OpenVLA | 7B | ~300ms | 3.3 | A100 |
| π₀ | 3B | ~50ms (5 denoise steps) | 20 | A100 |
| SmolVLA | 450M | ~33-66ms | 15-30 | RTX 4090 |
| Octo | 93M | ~20ms | 50 | A100 |
| ProgVLA | 100M | ~25ms | 40 | A100 |
| Evo-1 | ~1B | ~40ms | 25 | A100 |

**Key Insight:** There's a massive gap between large VLAs (3-300ms) and what's needed for reactive control (50-100Hz). Models >7B are impractical for real-time control.

---

## 5. Key Failure Patterns Across All Benchmarks

1. **Long-horizon fragility** — ALL models show 5-15% lower SR on long-horizon tasks
2. **Memorization over understanding** — LIBERO-PRO showed >90% SR models collapse to 0% under perturbation
3. **Camera sensitivity** — models overfit to fixed viewpoints
4. **Precision tasks** — insertion, peg-in-hole consistently hardest
5. **Distribution shift** — novel objects, positions, instructions cause catastrophic failure
6. **Language blindness** — models ignore instructions, rely on visual patterns

### Opportunity Areas
- **Robustness under perturbation** — biggest gap (90% → 0% on LIBERO-PRO)
- **Long-horizon planning** — 5-15% gap vs short tasks
- **Real-time inference** — need <20ms for reactive control
- **Sample efficiency** — most VLAs need 50+ demos per task

---

*Searches 6-10 results incorporated from web research*
