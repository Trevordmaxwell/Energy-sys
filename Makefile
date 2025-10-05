install:
bash scripts/setup_env.sh

format:
python -m pip install ruff black
ruff check --fix .
black .

test:
pytest -q

train-cpu:
bash scripts/train_small_cpu.sh

eval:
bash scripts/eval_small.sh
