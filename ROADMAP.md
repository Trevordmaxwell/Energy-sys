# UELM Roadmap

## Phase 0 — Boot (Mac CPU only)
- Target tiny configuration (`d=256`, `K=2048`, `len=512`).
- Use Anderson solver with tolerance `1e-3` and 10–15 iteration cap.
- Deliverables: unit tests passing, CPU training reaches stable perplexity, simple generation works.

## Phase 1 — Stability & Speed
- Add damping schedules, β annealing, and line search safeguards.
- Produce convergence plots in evaluation scripts.
- Run tiny smoke training (100–200 steps) in CI.

## Phase 2 — Memory Scale-Up (CPU first)
- Provide optional approximate retrieval (Annoy/HNSW) behind configuration flags.
- Add rolling local memory compression utilities.

## Phase 3 — MPS & Occasional GPU
- Implement device selection with dtype safety for MPS and CUDA.
- Investigate mixed precision once stability is verified.

## Phase 4 — Evals & Interpretability
- Implement attention-equivalence diagnostics and energy decomposition per token.
- Study dependence on external memory banks and solver iterations.
