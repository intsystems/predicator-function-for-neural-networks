#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES="7"

CONFIG="configs/surrogate_hp_CIFAR100.json"

echo
echo "🚀 Начинаем выполнение скрипта train_models.py"
echo "   Конфигурация: $CONFIG"
echo "   Доступные GPU: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo

start_time=$(date +%s)

python train_models.py --hyperparameters_json "$CONFIG"

end_time=$(date +%s)

elapsed=$((end_time - start_time))

hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo
echo "✅ Выполнение завершено!"
echo
printf "⏱️  Общее время выполнения: %02d:%02d:%02d\n" $hours $minutes $seconds
echo
echo "----------------------------------------"
echo
