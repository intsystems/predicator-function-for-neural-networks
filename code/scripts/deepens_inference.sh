#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_experiments() {
    local CONFIG="$1"
    local DATASET="$2"

    echo "=== Running DARTS baseline for $DATASET ==="
    for i in {1..2}; do
        python DARTS_baseline.py --hyperparameters_json "$CONFIG"
    done
    mv best_models "best_models_${DATASET}_deepens"

    echo "=== Running inference surrogate for $DATASET ==="
    for i in {1..2}; do
        python inference_surrogate.py --hyperparameters_json "$CONFIG"
    done
    mv best_models "best_models_${DATASET}_random"
}

run_experiments "surrogate_hp_fashionmnist.json" "fashionmnist"
run_experiments "surrogate_hp_CIFAR10.json"      "CIFAR10"
run_experiments "surrogate_hp_CIFAR100.json"     "CIFAR100"
