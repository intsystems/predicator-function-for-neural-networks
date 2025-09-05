#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="5, 6, 7"

CONFIG="surrogate_hp_dev.json"
python train_models.py --hyperparameters_json "$CONFIG"
