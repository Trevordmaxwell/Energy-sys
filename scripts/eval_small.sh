#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m uelm.train.eval_perplexity --config-name small_cpu
echo "✅ Evaluation complete"
