# Project Status

## MVP Checklist

- [ ] Naive Hopfield retrieval (exact matmul, CPU)
- [ ] Piecewise-linear control path from embeddings
- [ ] Anderson solver with damping and energy-decrease checks
- [ ] Implicit differentiation via conjugate gradients
- [ ] Causal masking across memory and CDE couplings
- [ ] Tiny-Shakespeare training loop achieving target perplexity on CPU
- [ ] Smoke tests for shapes, causality, and convergence

## In Progress

- [ ] Approximate memory retrieval backends (Annoy/HNSW)
- [ ] Apple Silicon (MPS) configuration and performance notes
- [ ] Convergence dashboard with iteration and energy plots

## Known Limitations

- [ ] Long contexts currently require manual chunk compression
- [ ] Solver may restart on repetitive prompts (β schedule tuning pending)
