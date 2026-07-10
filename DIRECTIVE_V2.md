# DIRECTIVE v2 — Revised Workstream Plan
*Updated: 2026-07-10 10:51 GMT+8*

## Venue Target: RA-L / ICRA 2027
- Sim-only results insufficient → plan real-robot validation (LeRobot SO-101 class)
- Timeline must include hardware experiments

---

## Workstream 1: PerturbVLA → "Closing the LIBERO-PRO Collapse"
**Status:** 🟡 ACTIVE — repositioning after being scooped

### SCOOPED By:
- **RobustVLA** (2025, github.com/gakakulicc/RobustVLA) — multi-modal perturbation + consistency training
- **RoVLA** (2026, LIBERO-Plus) — similar approach

### New Scope (Analysis-First Framing):
1. **Reproduce baselines**: RobustVLA and RoVLA numbers on LIBERO-PRO
2. **Identify gap**: Their +10-12% improvement leaves headroom in memorization-collapse regime
   - Specifically: object replacement, corrupted instructions
3. **Diagnosis agent**: Run intervention experiments answering WHY the model memorizes
   - Which components ignore vision/language inputs?
   - When does the model stop attending to observations?
4. **Fix derived from diagnosis**: Not a recipe, but an intervention motivated by the analysis

### Mandatory Deliverables:
- [ ] RobustVLA reproduced on LIBERO-PRO
- [ ] RoVLA reproduced on LIBERO-PRO
- [ ] Intervention analysis (vision/language ablation)
- [ ] Diagnosis-derived fix with ablation
- [ ] Real-robot validation on SO-101

---

## Workstream 2: MultiRes-Action
**Status:** 🔴 BLOCKED — design-space audit required before GPU spend

### Blocking Audit: Compare against:
- [ ] Helix (Figure AI) — dual-system architecture
- [ ] GR00T N1 (NVIDIA) — dual-head design
- [ ] RT-H (Google) — hierarchical actions
- [ ] π₀.₅ (Physical Intelligence) — dual-system (high-level + low-level)

### Audit Question:
What, precisely, is different about MultiRes-Action's coarse-to-fine design vs. these existing dual-head/hierarchical approaches?

### Gate:
Only proceeds to GPU training if audit finds a defensible novelty claim.

---

## Workstream 3: MambaFlow
**Status:** ⚪ DEPRIORITIZED — gated on budget + pilot

### Conditions to Proceed:
1. Budget remains after WS1 and WS2 complete
2. 10%-scale pilot run demonstrates training stability
3. Must differentiate from **StreamingVLA** (2026, action flow matching for latency)

---

## Infrastructure Rules (Unchanged from v1):
- Every metric → raw log + timestamp + checkpoint hash
- Run manifest on every training job
- Branch per agent; merge to main only with manifest
- Paper agent: no numbers without skeptic sign-off (\TODO{} placeholders)
- Real-robot validation required for venue target

## Status Table Format:
| Agent | Job | GPU | Last Checkpoint | ETA |
|-------|-----|-----|-----------------|-----|
| (update at every sync point) |
