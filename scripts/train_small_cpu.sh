#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m uelm.train.train_lm \
  --config-name small_cpu \
  train.max_steps=1500 train.log_every=50 \
  train.solver.max_iter=12 \
  data=tiny_shakespeare \
  model=small_cpu
