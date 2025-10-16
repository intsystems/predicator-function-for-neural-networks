#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES="1,2,3"

CONFIG="surrogate_hp_dev.json"

python train_surrogate.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"

python train_models.py --hyperparameters_json "$CONFIG"

echo "=== Все этапы успешно завершены ==="
