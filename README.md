# Unified Equilibrium Language Model (UELM)

## What is UELM?

The Unified Equilibrium Language Model solves for a single hidden state that balances a smooth Neural CDE flow with associative Modern Hopfield memory inside one energy function. At equilibrium these forces agree, enabling next-token prediction with attention-like behavior and implicit infinite depth.

## Why you might want this

- Preserves attention-style retrieval while introducing global consistency.
- Uses an equilibrium solver to emulate very deep networks without explicit stacking.
- Designed to run research demos on modest Mac CPUs while remaining extensible.

## Hardware assumptions

- **Default:** macOS CPU with 8 GB RAM.
- **Optional:** Apple Silicon (Metal/MPS) or an occasional CUDA GPU via configuration overrides.

## Quickstart (CPU, ~10–20 min)

```bash
# 1) Create venv and install CPU dependencies
bash scripts/setup_env.sh

# 2) Grab a tiny dataset
bash scripts/download_tiny.sh

# 3) Train a tiny model on CPU
bash scripts/train_small_cpu.sh

# 4) Evaluate perplexity and solver convergence
bash scripts/eval_small.sh

# 5) Generate a few tokens
python -m uelm.train.infer --config-name small_cpu prompt="To be, or not to be"
```

## What’s implemented vs. TODO

See [STATUS.md](STATUS.md) for the current feature checklist.

## Docs map

- `docs/OVERVIEW.md`: plain-language explainer
- `docs/TECHNICAL_DESIGN.md` and `docs/MATH_DETAILS.md`: deeper dives
- `docs/SOLVER_NOTES.md`: convergence and stability techniques
- `docs/CPU_TIPS.md`: tips for working within an 8 GB budget
- `docs/EVALS.md`: evaluation protocols

## Citing & License

The project uses the MIT License. A `CITATION.cff` file is included for reference.
