[![Test Status](https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg)](https://github.com/intsystems/ProjectTemplate/tree/master)
[![Coverage](https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master)](https://app.codecov.io/gh/intsystems/ProjectTemplate)
[![Docs](https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg)](https://intsystems.github.io/ProjectTemplate/)

## Название исследуемой задачи

**Использование предикаторной функции для построения ансамбля нейросетей**  
**Тип научной работы**: M1P  
**Автор**: Уденеев Александр Владимирович  
**Научный руководитель**: Бахтеев Олег  
**Научный консультант (при наличии)**: Бабкин Петр

---

## Abstract

The automated search for optimal neural network architectures (NAS) is a challenging computational problem, and Neural Ensemble Search (NES) is even more complex.  
In this work, we propose a surrogate-based approach for ensemble creation. Neural architectures are represented as graphs, and their predictions on a dataset serve as training data for the surrogate function.  
Using this function, we develop an efficient NES framework that enables the selection of diverse and high-performing architectures. The resulting ensemble achieves superior predictive accuracy on CIFAR-10 compared to other one-shot NES methods, demonstrating the effectiveness of our approach.

**Keywords**: NES, GCN, triplet loss, surrogate function

---

## Research publications

1. _To be added_

---

## Presentations at conferences on the topic of research

1. _To be added_

---

# NAS with Diversity using Surrogate Models

## Обзор проекта
Этот проект реализует нейроэволюционный поиск архитектур (NAS) с акцентом на разнообразие моделей в ансамбле. Система состоит из 3 основных компонентов:

1. **Обучение моделей** (`train_models.py`) - обучение ансамбля моделей и оценка их качества
2. **Обучение суррогатных моделей** (`train_surrogate.py`) - создание суррогатных моделей для предсказания качества и разнообразия архитектур
3. **Поиск архитектур** (`inference_surrogate.py`) - поиск оптимальных архитектур с помощью суррогатных моделей

### Общие гиперпараметры:
```json
{
    "seed":42,   // Сид для воспроизводимости
    "num_workers": 4, // Количество ядер процессора для загрузки данных
    "device": "cpu", // Устройство для вычислений
    "developer_mode": true, // Режим разработчика (в нем модели обучаются лишь на одном батче)
}
```
Логи всех модулей сохраняются в папку logs.

## 1. train_models.py
Скрипт для обучения ансамбля моделей и оценки их качества.

### Основные функции:
- Обучение множества моделей с разными архитектурами
- Оценка индивидуальной точности каждой модели
- Оценка точности и калибровки ансамбля моделей
- Сохранение результатов обучения

### Гиперпараметры:
```json
{
    "n_models_to_evaluate": 100,        // Количество моделей для генерации, если подготавливаем датасет, иначе ни на что не влияет
    "evaluate_ensemble_flag": false,     // Флаг оценки ансамбля (true/false). Если true, то оцениваем ансамбль, если false, то подготавливаем датасет
    "best_models_save_path": "best_models/", // Путь к архитектурам моделей, из которых составляется ансамбль
    "dataset_name": "CIFAR10",           // Используемый датасет (CIFAR10/CIFAR100/FashionMNIST)
    "final_dataset_path": "datasets/final_dataset/", // Путь к папке, куда будем скачивать датасеты для обучения моделей
    "n_epochs_final": 1,                 // Количество эпох обучения
    "lr_start_final": 0.025,             // Начальный шаг обучения
    "lr_end_final": 1e-3,                // Конечный шаг обучения
    "batch_size_final": 96,              // Размер батча
    "width": 4,                          // Ширина слоев в DARTS
    "num_cells": 3,                      // Количество ячеек в DARTS
    "n_ece_bins": 15                     // Количество бинов для расчета ECE
}
```

### Процесс работы:
1. Загрузка или генерация архитектур моделей
2. Создание DataLoader'ов для выбранного датасета
3. Обучение каждой модели:
   - Инициализация архитектуры DARTS
   - Обучение
   - Сохранение результатов
4. При флаге `evaluate_ensemble_flag`:
   - Оценка ансамбля на тестовых данных
   - Расчет точности и ECE
   - Сохранение результатов оценки

## 2. train_surrogate.py
Скрипт для обучения суррогатных моделей, предсказывающих качество и разнообразие архитектур.

### Основные функции:
- Загрузка датасета с архитектурами и результатами
- Расчет матрицы разнообразия
- Преобразование архитектур в графы
- Обучение GAT-моделей для предсказания:
  - Точности архитектуры
  - Эмбеддингов разнообразия

### Гиперпараметры:
```json
{
    "dataset_path": "datasets/cifar100_archs", // Путь к датасету архитектур моделей на которых будут обучаться суррогатные функции
    "n_models": 300,                     // Количество используемых моделей (не больше, чем в датасете)
    "diversity_matrix_metric": "overlap", // Метрика разнообразия (overlap/js)
    "upper_margin": 0.75,                // Верхний квантиль для дискретизации матрицы похожести
    "lower_margin": 0.25,                // Нижний квантиль для дискретизации матрицы похожести
    "input_dim": 8,                      // Размерность признаков
    "acc_num_epochs": 10,                // Количество эпох обучения модели точности
    "acc_lr_start": 1e-2,                // Начальный шаг обучения для модели точности
    "acc_lr_end": 1e-5,                  // Конечный шаг обучения для модели точности
    "acc_dropout": 0.2,                  // Dropout для модели точности
    "acc_n_heads": 16,                   // Количество голов в модели точности
    "div_num_epochs": 5,                 // Количество эпох обучения модели разнообразия
    "div_lr_start": 1e-3,                // Начальный шаг обучения для модели разнообразия
    "div_lr_end": 1e-6,                  // eta_min для Cosine scheduler
    "div_dropout": 0.1,                  // Dropout для модели разнообразия
    "div_n_heads": 4,                    // Количество голов в модели разнообразия
    "margin": 1,                         // Отступ для triplet loss
    "div_output_dim": 128,               // Размерность эмбеддинга разнообразия
    "surrogate_inference_path": "surrogate_models/", // Путь для сохранения суррогатных моделей
    "train_size": 0.8,                   // Размер тренировочной выборки
    "batch_size": 8,                     // Размер батча
}
```

### Процесс работы:
1. Загрузка датасета с архитектурами и результатами
2. Расчет матрицы разнообразия между моделями
3. Преобразование матрицы в дискретный вид
4. Преобразование архитектур в графовые представления
5. Создание датасетов для обучения:
   - Для предсказания точности
   - Для обучения эмбеддингов разнообразия (триплеты)
6. Обучение двух GAT-моделей:
   - Модель точности (регрессия)
   - Модель разнообразия (эмбеддинги)
7. Сохранение обученных моделей

## 3. inference_surrogate.py
Скрипт для поиска оптимальных архитектур с помощью обученных суррогатных моделей.

### Основные функции:
- Инициализация обученных суррогатных моделей
- Генерация новых архитектур
- Предсказание точности и эмбеддингов
- Отбор архитектур по точности и разнообразию
- Кластеризация архитектур и выбор представителей
- Визуализация результатов
- Сохранение лучших архитектур

### Гиперпараметры:
```json
{
    "n_ensemble_models": 2,                           // Количество моделей в ансамбле
    "n_models_in_pool": 128,                          // Размер пула кандидатов
    "n_models_to_generate": 4096,                     // Количество генерируемых архитектур
    "min_accuracy_for_pool": 0.01,                    // Минимальная точность для попадания в пул
    "acc_distance_gamma":0.5,                         // Коэффициент gamma для расчёта score: (1 - gamma) * acc + gamma * dist
    "min_acc_and_div_to_ensemble": 0.01,              // Трешхолд score
    "random_choice_out_of_best" : false,              // Выборием случайные архитектуры прошедшие трешхолд по точности
    "greedy_choice_out_of_best" : true,               // Эволюционный алгоритм, выбираем лучше архитектуры по score: (1 - gamma) * acc + gamma * dist
    "no_clusters_choice": false,                      // Жадно выбираем по score: (1 - gamma) * acc + gamma * dist модели из сгенерированных
    "best_models_save_path": "best_models/",          // Путь для сохранения лучших архитектур
    "tmp_archs_path": "datasets/tmp_archs/",          // Временное хранилище для генерируемых архитектур
    "prepared_dataset_path": "datasets/cifar100_archs", // Путь к уже обученным моделям(веса должны лежать по тому же пути,
    //  но иметь постфикс _pth вместо _archs)
}
```
Если не выбран ни один из методов генерации ансамбля, то ансамбль генерируется из моделей, соответствующих центроидам эмбеддингов, где кластеризация происходит среди моделей прошедших трешхолд по точности.


### Процесс работы:
1. Загрузка обученных суррогатных моделей
2. Генерация архитектур:
   - Генерация большого количества случайных архитектур
   - Предсказание их точности и эмбеддингов разнообразия
   - Фильтрация по минимальной точности
3. Формирование пула кандидатов:
   - Постепенное заполнение пула лучшими архитектурами
   - Отбор по максимальному расстоянию в пространстве эмбеддингов
4. Кластеризация:
   - Кластеризация архитектур в пуле
   - Выбор наиболее репрезентативных моделей из каждого кластера
5. Визуализация (при включенном флаге):
   - PCA + t-SNE для визуализации пространства эмбеддингов
   - Отображение кластеров и выбранных моделей
6. Сохранение лучших архитектур

## Инференс суррогатных функций
```bash
# Перед запуском необходимо скачать выставить флаг "evaluate_ensemble_flag": true
./start_all.sh
```

## Запуск системы

1. **Подготовка датасета:** 
```bash
# Перед запуском необходимо выставить флаг "evaluate_ensemble_flag": false и указать количество моделей 
# для оценки
python train_models.py --hyperparameters_json surrogate_hp.json 
```

2. **Обучение суррогатных моделей:**
```bash
python train_surrogate.py --hyperparameters_json surrogate_hp.json
```

3. **Поиск архитектур:**
```bash
python inference_surrogate.py --hyperparameters_json surrogate_hp.json
```

4. **Оценка ансамбля:**
```bash
# В файле конфигурации установить "evaluate_ensemble_flag": true
python train_models.py --hyperparameters_json surrogate_hp.json
```


## Требования
- Python 3.8+
- PyTorch 1.10+
- torchvision
- scikit-learn
- tqdm
- NNI (Neural Network Intelligence)