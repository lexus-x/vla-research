# Audit Agent — MultiRes-Action Design Space Review
## BLOCKING GATE: Must complete before any GPU spend

**Branch:** agent/audit-multires
**Question:** What, precisely, is different about MultiRes-Action vs. existing dual-head/hierarchical VLA architectures?

---

## Models to Compare

### 1. Helix (Figure AI, 2025)
- **Architecture:** Dual-system — fast reactive system + slow deliberative system
- **Key claim:** Fast system handles low-level motor control; slow system handles planning
- **Overlap with MultiRes-Action:** Hierarchical decomposition of control
- **Key difference to investigate:** Is MultiRes-Action's "coarse planner + fine controller" just Helix renamed?

### 2. GR00T N1 (NVIDIA, 2025)
- **Architecture:** Dual-system — System 1 (fast, reactive, action generation) + System 2 (slow, VLM-based reasoning)
- **Key claim:** System 2 interprets instructions and plans; System 1 executes
- **Overlap with MultiRes-Action:** Two-level hierarchy with different speeds
- **Key difference to investigate:** GR00T N1's System 2 is a full VLM; MultiRes-Action's "coarse planner" is flow matching. Is the distinction meaningful or cosmetic?

### 3. RT-H (Google, 2024)
- **Architecture:** Hierarchical — high-level "language actions" (subgoals in language space) + low-level motor actions
- **Key claim:** Intermediate language representation bridges instruction and motor control
- **Overlap with MultiRes-Action:** Hierarchical action decomposition
- **Key difference to investigate:** RT-H uses language as the intermediate representation; MultiRes-Action uses continuous latent actions. Does this matter?

### 4. π₀.₅ (Physical Intelligence, 2025)
- **Architecture:** Dual-system — high-level semantic reasoning (chain-of-thought) + low-level flow matching action expert
- **Key claim:** High-level reasoning about "what to do next"; low-level generates motor commands
- **Overlap with MultiRes-Action:** Two-level with flow matching at the low level
- **Key difference to investigate:** π₀.₅'s high-level is language-based CoT; MultiRes-Action's coarse planner is a learned latent. Is this a real distinction?

---

## Audit Matrix

| Dimension | Helix | GR00T N1 | RT-H | π₀.₅ | MultiRes-Action |
|-----------|-------|----------|------|-------|-----------------|
| High-level representation | VLM (fast+slow) | VLM (System 2) | Language motions | CoT reasoning | Flow matching (coarse) |
| Low-level representation | Action tokens | Action tokens (System 1) | Motor actions conditioned on lang | Flow matching | MLP (fine) |
| Intermediate abstraction | Latent (undisclosed) | Latent between systems | Language phrases ("move arm forward") | Subtask descriptions | Learned latent action |
| Training signal | End-to-end (undisclosed) | End-to-end | Language supervision for intermediate | End-to-end + web data co-train | Hierarchical loss (coarse+fine) |
| Inference speed | Undisclosed | Fast (System 1) + Slow (System 2) | Single-pass (language pred → action) | 10-20 denoising steps | 2-pass (coarse then fine) |
| Cross-embodiment | No (Figure 02 only) | Yes (humanoid) | Limited (Google robots) | Yes (multi-robot) | Planned |
| Open-source | No | Yes (NVIDIA) | No | Partial (openpi) | Planned |

**Key findings:**
1. RT-H uses language as the intermediate representation; MultiRes-Action uses a learned latent. This is a real architectural difference.
2. GR00T N1 and Helix are dual-system but undisclosed internals; MultiRes-Action's coarse-to-fine is explicit.
3. π₀.₅'s high-level is CoT (language-based); MultiRes-Action's coarse planner is flow matching (continuous). Different modality.
4. The closest prior work is RT-H — but RT-H requires language supervision for the intermediate level; MultiRes-Action learns it end-to-end.

**Preliminary assessment:** There IS a defensible novelty claim in the intermediate representation (learned latent vs. language), but it's narrow. The audit should focus on whether this distinction matters empirically.

---

## Defensibility Criteria

MultiRes-Action is **publishable** if and only if at least ONE of these is true:

1. **Different intermediate representation**: The coarse-to-fine decomposition uses a representation that no prior work uses (e.g., learned latent actions vs. language vs. VLM features)

2. **Different training paradigm**: The way the two levels are trained is fundamentally different (e.g., separate vs. joint, different losses)

3. **Empirical gap**: Even if the architecture is similar, MultiRes-Action achieves measurably better results on a specific, well-defined task regime (e.g., long-horizon tasks where existing dual-head methods plateau)

4. **Efficiency advantage**: MultiRes-Action achieves comparable performance with significantly fewer parameters or faster inference

**If NONE of these hold, MultiRes-Action should be abandoned or merged with an existing approach.**

---

## Deliverable

A 2-page report with:
1. Architecture comparison table (filled in)
2. Specific novelty claims (if any)
3. Recommendation: PROCEED / PIVOT / ABANDON
4. If PROCEED: what specific experiments demonstrate the difference

---

## Status: 🔴 IN PROGRESS — searching for papers
