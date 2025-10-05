# Agent Task Graph

## T1: Scaffolding
- Create directory structure and empty modules under `uelm/core`, `uelm/models`, `uelm/train`.
- Acceptance: `pytest -k scaffolding` (to be implemented) should pass.

## T2: Hopfield (Exact CPU)
- Implement `HopfieldMemory` with softmax retrieval and causal masking.
- Acceptance: `pytest uelm/uelm/tests/test_memory.py::test_attention_equivalence`.

## T3: Piecewise-Linear CDE Path
- Build `build_piecewise_linear_path` and a small causal vector field.
- Acceptance: `pytest uelm/uelm/tests/test_energy.py::test_cde_shapes`.

## T4: Energy Builder
- Implement `total_energy()` combining CDE residuals, Hopfield term, and regularizers.
- Acceptance: `pytest uelm/uelm/tests/test_energy.py::test_energy_gradients`.

## T5: Anderson Solver
- Implement damped acceleration with energy-decrease fallback.
- Acceptance: `pytest uelm/uelm/tests/test_solver.py::test_convergence_on_convex`.

## T6: Implicit Differentiation
- Provide conjugate-gradient based backward pass.
- Acceptance: `pytest uelm/uelm/tests/test_training_step.py`.

## T7: Training Loop
- Implement CPU-friendly trainer with checkpointing and logging.
- Acceptance: perplexity drop ≥10% on Tiny Shakespeare in ≤600 steps.

## T8: Generation
- Implement autoregressive generation with equilibrium warm-starts.
- Acceptance: Generate 50 tokens without error using `python -m uelm.train.infer`.

## T9: Approximate Retrieval (Optional)
- Integrate Annoy/HNSW backends under configuration flags.
- Acceptance: `pytest uelm/uelm/tests/test_memory.py::test_backend_consistency`.

## T10: MPS/GPU Configs
- Add device/dtype options and document in `docs/CPU_TIPS.md`.
- Acceptance: Manual smoke run on MPS (documented by maintainer).
