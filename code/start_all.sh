#!/usr/bin/env bash
set -euo pipefail

CONFIG="surrogate_hp.json"

echo "=== Запуск surrogate_train.py ==="
python train_surrogate.py --hyperparameters_json "$CONFIG"

echo "=== Запуск inference_surrogate.py ==="

python inference_surrogate.py --hyperparameters_json "$CONFIG"
python train_models.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"
python train_models.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"
python train_models.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"
python train_models.py --hyperparameters_json "$CONFIG"

python inference_surrogate.py --hyperparameters_json "$CONFIG"
python train_models.py --hyperparameters_json "$CONFIG"
# echo "=== Все этапы успешно завершены ==="
