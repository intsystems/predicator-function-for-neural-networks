#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="1,2,3"

CONFIG="surrogate_hp_dev.json"

echo
echo "🚀 Начинаем выполнение скрипта train_models.py"
echo "   Конфигурация: $CONFIG"
echo "   Доступные GPU: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo

# Запоминаем время начала
start_time=$(date +%s)

# Запускаем Python-скрипт
python train_models.py --hyperparameters_json "$CONFIG"

# Запоминаем время окончания
end_time=$(date +%s)

# Считаем разницу
elapsed=$((end_time - start_time))

# Форматируем в часы, минуты, секунды
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

# Красивый вывод
echo
echo "✅ Выполнение завершено!"
echo
printf "⏱️  Общее время выполнения: %02d:%02d:%02d\n" $hours $minutes $seconds
echo
echo "----------------------------------------"
echo
