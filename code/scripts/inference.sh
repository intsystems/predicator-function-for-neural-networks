#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/surrogate_hp_CIFAR100.json"

python inference_surrogate.py --hyperparameters_json "$CONFIG"

