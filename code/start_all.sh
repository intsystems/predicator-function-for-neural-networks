#!/usr/bin/env bash
set -euo pipefail

CONFIG="surrogate_hp.json"

echo "=== Запуск surrogate_train.py ==="
python3 train_surrogate.py --hyperparameters_json "$CONFIG"

echo "=== Запуск inference_surrogate.py ==="
python3 inference_surrogate.py --hyperparameters_json "$CONFIG"

echo "=== Запуск train_models.py ==="
python3 train_models.py --hyperparameters_json "$CONFIG"

echo "=== Все этапы успешно завершены ==="
