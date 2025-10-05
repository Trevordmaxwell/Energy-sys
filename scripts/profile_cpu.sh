#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m uelm.train.profile --config-name small_cpu
