#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/surrogate_hp_CIFAR100.json"
STEP_SIZE=100
MAX_MODELS=1000

PYTHONPATH=. python junk/ablation_study.py \
    --hyperparameters_json "$CONFIG" \
    --step_size "$STEP_SIZE" \
    --max_models "$MAX_MODELS" \
    --output_base /home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/junk/ablation_results \
    --plots_dir /home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/junk/ablation_plots \
    --strategies ucb_bayes random greedy  \
    --ucb_passes 20 \
    --ucb_beta 1.0 \
    --skip_existing
