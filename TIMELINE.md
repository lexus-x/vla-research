# VLA Research Pipeline - 3 Hour Timeline

## Phase 1: RESEARCH & IDEA GENERATION (0:00 - 0:30) ← NOW
- [x] Spawn 5 parallel research agents
- [ ] Agent 1: VLA Paper Survey → 01_vla_survey.md
- [ ] Agent 2: Cross-Domain Mining → 02_cross_domain_ideas.md
- [ ] Agent 3: Benchmark Analysis → 03_benchmark_analysis.md
- [ ] Agent 4: Architecture Matrix → 04_architecture_matrix.md
- [ ] Agent 5: Novelty Database → 05_novelty_database.md

## Phase 2: IDEA SYNTHESIS & GATE FILTER (0:30 - 0:50)
- [ ] Synthesize all agent outputs
- [ ] Generate 10-15 candidate ideas
- [ ] Run each through CONSTRAINT_GATES.md
- [ ] Score and rank
- [ ] Select TOP 3

## Phase 3: IMPLEMENTATION (0:50 - 2:00)
- [ ] TOP 3 ideas → parallel implementation
- [ ] Each gets: code skeleton + training script + eval script
- [ ] Launch on AWS L40S (1 instance per idea)
- [ ] Monitor training, collect results

## Phase 4: EVALUATION & ABLATION (2:00 - 2:30)
- [ ] Run all 3 on LIBERO, LIBERO+, MetaWorld
- [ ] 3 seeds each
- [ ] Ablation studies
- [ ] Statistical analysis

## Phase 5: PAPER WRITING (2:30 - 3:00)
- [ ] Select winner
- [ ] Write full LaTeX paper
- [ ] Generate figures and tables
- [ ] Push to GitHub
- [ ] Final review
