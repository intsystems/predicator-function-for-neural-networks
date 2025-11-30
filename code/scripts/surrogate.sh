#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


CONFIG="surrogate_hp_CIFAR10.json"
python train_surrogate.py --hyperparameters_json "$CONFIG"
