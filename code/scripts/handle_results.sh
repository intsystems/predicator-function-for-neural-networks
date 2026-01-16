#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../dependencies"

DATA_DIR="/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/fashionmnist_deepens"

python handle_results.py --data_dir "$DATA_DIR"    

