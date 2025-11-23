#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES="5,6,7"

CONFIG="surrogate_hp_CIFAR100.json"

python inference_surrogate.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"

# python inference_surrogate.py --hyperparameters_json "$CONFIG"

# python inference_surrogate.py --hyperparameters_json "$CONFIG"

# python train_models.py --hyperparameters_json "$CONFIG"

echo "=== Все этапы успешно завершены ==="
