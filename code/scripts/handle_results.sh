#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../dependencies"

DATA_DIR="/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/results/cifar10"

python handle_results.py --data_dir "$DATA_DIR"    

