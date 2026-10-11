#!/bin/bash
# Run from the project folder:  bash finish_now.sh
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results/logs results/tables

# 1. robustness run in the background (skips it if one is already running)
if pgrep -f "run_pipeline.py --basket A --robust" >/dev/null; then
  echo "[1] robustness run already running"
else
  [ -f results/logs/runs.jsonl ] && mv results/logs/runs.jsonl "results/logs/runs_partial_$(date +%H%M%S).jsonl"
  nohup caffeinate -i python experiments/run_pipeline.py --basket A --robust --robust-skip baseline --seeds 1 > results/logs/run_output_robust2.txt 2>&1 &
  echo "[1] robustness run started in background"
fi

# 2. tables for both baskets (main logs)
python experiments/make_tables.py results/logs/runs_A_expanding.jsonl > results/tables_A.txt 2>&1 && echo "[2] basket A tables -> results/tables_A.txt"
python experiments/make_tables.py results/logs/runs_B_expanding.jsonl > results/tables_B.txt 2>&1 && echo "[2] basket B tables -> results/tables_B.txt"

# 3. git: ignore big/local files, init, first commit
cat >> .gitignore <<'G'
.venv/
__pycache__/
*.npz
data/raw/
results/logs/run_output*.txt
results/logs/runs_partial_*
results/logs/*discard*
.DS_Store
G
sort -u .gitignore -o .gitignore
[ -d .git ] || git init -q
git add -A
git commit -q -m "Shriker modules S1-S14: data, labels, harness, classical models, stats" && echo "[3] first commit made"
git status --short | head -20
echo; echo "Done. Send me results/tables_A.txt and results/tables_B.txt if you want them checked."
