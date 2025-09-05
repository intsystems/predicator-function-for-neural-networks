#!/usr/bin/env bash
set -euo pipefail

CONFIG="surrogate_hp_dev.json"

python inference_surrogate.py --hyperparameters_json "$CONFIG"

