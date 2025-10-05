#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
if [ ! -f tiny_shakespeare.txt ]; then
  curl -L -o tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi
echo "✅ Downloaded tiny dataset"
