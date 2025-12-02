# Как запустить обучение

## 1. Подготовка директорий
1. Создать папку `best_models` внутри директории `code`.  
2. Если папка уже существует — **очистить её содержимое**.  
3. Переместить в неё все директории формата `models_json_#`.

---

## 2. Проверка конфигураций

Убедитесь, что в конфигурационных файлах заданы корректные параметры.

### Для датасета **FashionMNIST**:

```json
{
    "seed": 42,
    "use_pretrained_models_for_ensemble": false,
    "developer_mode": false,

    "train_size_final": 1.0,
    "n_epochs_final": 125,
    "lr_start_final": 0.025,
    "lr_end_final": 1e-3,
    "weight_decay": 3e-4,
    "auxiliary_loss_weight": 0.4,
    "batch_size_final": 96,
    "optimizer": "SGD",

    "dataset_name": "FashionMNIST",
    "max_per_gpu": 6,

    "final_dataset_path": "datasets/final_dataset/",
    "evaluate_ensemble_flag": true,
    "n_models_to_evaluate": 2000,
    "output_path": "output/",
    "n_ece_bins": 16,

    "num_cells": 3,
    "width": 16
}
````

---

### Для датасетов **CIFAR10** и **CIFAR100**:

```json
{
    "seed": 42,
    "use_pretrained_models_for_ensemble": false,
    "developer_mode": false,

    "train_size_final": 1.0,
    "n_epochs_final": 600,
    "lr_start_final": 0.025,
    "lr_end_final": 1e-3,
    "weight_decay": 3e-4,
    "optimizer": "SGD",
    "auxiliary_loss_weight": 0.4,
    "batch_size_final": 96,

    "dataset_name": "CIFAR100",
    "max_per_gpu": 5,

    "final_dataset_path": "datasets/final_dataset/",
    "evaluate_ensemble_flag": true,
    "n_models_to_evaluate": 2000,
    "output_path": "output/",
    "n_ece_bins": 16,

    "num_cells": 20,
    "width": 36
}
```

Если возникает OOM — уменьшите `max_per_gpu`, чтобы снизить число одновременно обучаемых моделей.

---

## 3. Запуск обучения

Перейдите в директорию `scripts` и откройте файл `train_models`.
Укажите в нём:

* путь к нужному конфигу;
* список GPU, которые должны использоваться.

После редактирования запустите скрипт из этой же директории.

* Обучение одного ансамбля **FashionMNIST** занимает около **2–3 часов**.
* Обучение ансамбля для **CIFAR10/CIFAR100** может занять **до 24 часов**.

