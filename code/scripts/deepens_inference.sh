#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_baseline() {
    local CONFIG="$1"
    local DATASET="$2"

    echo "=== Running DARTS baseline for $DATASET ==="
    python DARTS_baseline.py --hyperparameters_json "$CONFIG"

    local OUTDIR="best_models_${DATASET}_baseline"
    mkdir -p "$OUTDIR"
    mv best_models "$OUTDIR"
    echo "[OK] Saved to $OUTDIR"
}

# ──────────────────────────────────────────────
# Example usage:
# run_baseline surrogate_hp_fashionmnist.json fashionmnist
# run_baseline surrogate_hp_CIFAR10.json      CIFAR10
# run_baseline surrogate_hp_CIFAR100.json     CIFAR100
# ──────────────────────────────────────────────

run_baseline "surrogate_hp_fashionmnist.json" "fashionmnist"
