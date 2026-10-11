#!/bin/bash
# Run when the robustness run has finished:  bash finish_after_robust.sh
cd "$(dirname "$0")"
source .venv/bin/activate
if pgrep -f "run_pipeline.py --basket A --robust" >/dev/null; then echo "Still running - wait."; exit 1; fi
echo "tracebacks: $(grep -c Traceback results/logs/run_output_robust2.txt)"
cat results/logs/runs_A_robust_baseline.jsonl results/logs/runs.jsonl > results/logs/runs_A_robust_all.jsonl
python experiments/make_tables.py results/logs/runs_A_expanding.jsonl results/logs/runs_A_robust_all.jsonl > results/tables_A_robust.txt 2>&1
tail -n 30 results/tables_A_robust.txt
git add -A
git commit -q -m "Add robustness runs and result tables" && echo "committed"
