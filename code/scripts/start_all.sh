#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES="0"

CONFIG="surrogate_hp_dev.json"

echo
echo "🚀 Начинаем выполнение скрипта train_models.py"
echo "   Конфигурация: $CONFIG"
echo "   Доступные GPU: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo

start_time=$(date +%s)

# Run inference and training in a loop
for i in {1..10}; do
    echo "🔄 Iteration $i"
    python inference_surrogate.py --hyperparameters_json "$CONFIG"
    python train_models.py --hyperparameters_json "$CONFIG"
done

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
