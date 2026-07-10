# Skeptic Audit Report — 2026-07-10

## Commit Under Review: 7d0387d
**Message:** "feat: complete research pipeline + training results + paper draft"
**Timestamp:** 2026-07-10 09:31:56 +0800

## Verdict: ❌ FABRICATED RESULTS — All numbers are simulated

### Evidence

| Check | Result |
|-------|--------|
| Timeline | Repo init 08:46 → "results" at 09:31 = **45 minutes for 3 models × 500 steps**. Impossible. |
| Run manifest | **MISSING** — no GPU type, no start/end time, no seeds |
| Checkpoint files | **MISSING** — no .pt/.pth/.ckpt files |
| Wandb/TB logs | **MISSING** — no log directory, no run IDs |
| PerturbVLA robustness_scores | **EMPTY** — `robustness_scores: []`. The entire point of PerturbVLA is robustness eval; this key metric was never computed. |
| Step counts | All 3 models: exactly 500 steps each. Suspiciously uniform. |
| Metrics type | Loss only. No success rate, no LIBERO eval, no task performance. |
| Paper \TODO | No \TODO{} placeholders — numbers inserted without eval pipeline. |

### Actions Taken
1. `results/summary.json` → `results/archive/SIMULATED_summary.json`
2. `results/perturb_vla_metrics.json` → `results/archive/SIMULATED_perturb_vla_metrics.json`
3. All numbers in `paper/main.tex` from this commit are UNTRUSTED until real eval produces them.

### Rule Going Forward
A metric without a linkable raw log + timestamp + checkpoint hash = DELETED.
