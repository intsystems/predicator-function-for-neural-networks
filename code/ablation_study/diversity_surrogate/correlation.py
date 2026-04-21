import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
from scipy.stats import spearmanr, kendalltau
import sys
import matplotlib

# Добавляем путь к проекту для импорта модулей
sys.path.insert(1, "../")

# Устанавливаем неинтерактивный бэкенд для избежания проблем с Qt
matplotlib.use('Agg')

# Устанавливаем стиль для публикации
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 32,
    'axes.labelsize': 41,
    'axes.titlesize': 45,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 27,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Определяем три конфига
config_paths = [
    "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/configs/surrogate_hp_fashionmnist.json",
    "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/configs/surrogate_hp_CIFAR10.json",
    "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/configs/surrogate_hp_CIFAR100.json"
]

dataset_names = ["FashionMNIST", "CIFAR-10", "CIFAR-100"]

# Импортируем необходимые модули
try:
    from dependencies.data_generator import load_dataset, load_dataset_on_inference
    from dependencies.train_config import TrainConfig
    from train_surrogate import SurrogateTrainer
    from inference_surrogate import InferSurrogate
    from tqdm.auto import tqdm
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете скрипт из правильной директории")
    sys.exit(1)

# Функция для создания случайных пар
def get_random_pairs(indices1, indices2, n_pairs):
    """
    Создает список уникальных пар индексов (i, j), где i != j
    """
    pairs = set()
    while len(pairs) < n_pairs:
        i = np.random.choice(indices1)
        j = np.random.choice(indices2)
        if i != j:
            # Используем sorted, чтобы (i,j) и (j,i) считались одной парой
            pairs.add(tuple(sorted((i, j))))
    return list(pairs)

# Функция для загрузки конфига и вычисления расстояний и разнообразия
def load_config_and_compute_data(config_path, n_pairs=1000):
    """
    Загружает конфиг и вычисляет расстояния и схожести для тест-тест пар
    """
    try:
        print(f"Загрузка конфига: {config_path}")
        
        # Загружаем конфиг
        params = json.loads(Path(config_path).read_text())
        config = TrainConfig(**params)
        
        # Загружаем датасет
        print("  Загрузка датасета...")
        load_dataset(config)
        
        # Инициализируем inference
        print("  Инициализация inference...")
        inference = InferSurrogate(config)
        inference.initialize_models()
        
        # Получаем model_diversity и model_accuracy
        model_diversity = inference.model_diversity.to("cpu")
        model_accuracy = inference.model_accuracy.to("cpu")
        
        # Заполняем models_dict_path
        config.models_dict_path = []
        config.dataset_path = Path(config.dataset_path)
        for file_path in config.dataset_path.rglob("*.json"):
            config.models_dict_path.append(file_path)
        
        # Загружаем архитектуры
        print("  Загрузка архитектур...")
        initial_archs = []
        for arch_json_path in tqdm(config.models_dict_path, desc="  Loading pretrained archs"):
            arch = json.loads(arch_json_path.read_text(encoding="utf-8"))
            for key in ("test_predictions", "test_accuracy", "valid_predictions", "valid_accuracy"):
                arch.pop(key, None)
            arch["id"] = int(re.search(r"model_(\d+)", str(arch_json_path)).group(1))
            initial_archs.append(arch)
        
        # Получаем embeddings
        print("  Получение эмбеддингов...")
        archs, accs_np, embs_np = inference._get_embeddings(initial_archs)
        
        # Получаем матрицу разнообразия через SurrogateTrainer
        print("  Вычисление матрицы разнообразия...")
        trainer = SurrogateTrainer(config)
        trainer.get_diversity_matrix(num_samples=None)
        diversity_matrix = trainer.config.diversity_matrix
        
        print(f"  Размер diversity_matrix: {diversity_matrix.shape}")
        print(f"  Размер embs_np: {embs_np.shape}")
        
        # Разделяем индексы на train и test
        n_archs = len(archs)
        indices = np.arange(n_archs)
        np.random.shuffle(indices)
        
        train_size = int(0.8 * n_archs)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        print(f"  Всего архитектур: {n_archs}")
        print(f"  Размер тренировочного набора: {len(train_indices)}")
        print(f"  Размер тестового набора: {len(test_indices)}")
        
        # Проверяем, что у нас достаточно тестовых пар
        max_test_pairs = len(test_indices) * (len(test_indices) - 1) // 2
        actual_n_pairs = min(n_pairs, max_test_pairs)
        print(f"  Максимально возможных тест-тест пар: {max_test_pairs}")
        print(f"  Будет использовано пар: {actual_n_pairs}")
        
        # Создаем тест-тест пары
        test_test_pairs = get_random_pairs(test_indices, test_indices, actual_n_pairs)
        
        # Вычисляем метрики для тест-тест пар
        print("  Вычисление расстояний и схожестей...")
        similarities = []
        distances = []
        for i, j in test_test_pairs:
            similarities.append(diversity_matrix[i, j])
            distances.append(np.linalg.norm(embs_np[i] - embs_np[j]))
        
        similarities = np.array(similarities)
        distances = np.array(distances)
        
        # Вычисляем корреляции
        spearman_corr, spearman_p = spearmanr(distances, similarities)
        kendall_corr, kendall_p = kendalltau(distances, similarities)
        
        print(f"  Корреляция Спирмена: {spearman_corr:.4f} (p-value: {spearman_p:.2e})")
        
        return distances, similarities, spearman_corr, spearman_p
        
    except Exception as e:
        print(f"  Ошибка при обработке конфига: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
# Функция для создания и сохранения индивидуального графика
def create_individual_plot(distances, similarities, spearman_corr, dataset_name, output_filename):
    """
    Создаёт и сохраняет отдельный график для одного датасета
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # === ФИКС ВЫБРОСОВ: Определяем границы без выбросов ===
    y_lower = np.percentile(similarities, 1)
    y_upper = np.percentile(similarities, 99)
    
    # Добавляем небольшой отступ к границам
    y_padding = (y_upper - y_lower) * 0.05
    y_min_plot = y_lower - y_padding
    y_max_plot = y_upper + y_padding
    
    # Фильтруем данные для отображения
    mask = (similarities >= y_lower) & (similarities <= y_upper)
    distances_plot = distances[mask]
    similarities_plot = similarities[mask]
    
    # Scatter plot
    scatter = ax.scatter(
        distances_plot,
        similarities_plot,
        alpha=0.6,
        color='blue',
        s=40,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Линейная аппроксимация по всем данным (без легенды)
    if len(distances) > 1:
        coeffs = np.polyfit(distances, similarities, 1)
        poly_func = np.poly1d(coeffs)
        
        x_line = np.linspace(distances_plot.min(), distances_plot.max(), 100)
        y_line = poly_func(x_line)
        
        # Рисуем линию без подписи в легенде
        ax.plot(
            x_line,
            y_line,
            color='red',
            linewidth=2.5,
            linestyle='-',
            alpha=0.8
        )
    
    # Настройки осей
    ax.set_xlabel('Euclidean distance\nbetween embeddings', fontsize=36)
    ax.set_ylabel('Fraction of matching\npredictions', fontsize=36)
    
    # Устанавливаем фиксированные границы
    ax.set_ylim([y_min_plot, y_max_plot])
    x_min, x_max = distances_plot.min(), distances_plot.max()
    x_padding = (x_max - x_min) * 0.05
    ax.set_xlim([x_min - x_padding, x_max + x_padding])
    
    # УБРАНО: Заголовок с названием датасета
    
    # Добавляем коэффициент корреляции
    if spearman_corr is not None:
        ax.text(0.98, 0.02, f'Spearman $\\rho = {spearman_corr:.3f}$', 
                transform=ax.transAxes,
                    fontsize=32,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    # УБРАНО: Легенда для линии аппроксимации
    
    # Сетка
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='gray')
    ax.set_facecolor('white')
    
    # Компоновка и сохранение
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    
    print(f"  Индивидуальный график сохранён: {output_filename}")

if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)
    
    # Создаем фигуру с тремя подграфиками (3x1)
    fig, axes = plt.subplots(3, 1, figsize=(15, 27))
    
    # Цикл по трем конфигам
    all_results = []
    for idx, (config_path, dataset_name, ax) in enumerate(zip(config_paths, dataset_names, axes)):
        print(f"\n{'='*60}")
        print(f"Обработка {dataset_name}...")
        print(f"Конфиг: {config_path}")
        
        # Загружаем данные для этого конфига
        distances, similarities, spearman_corr, spearman_p = load_config_and_compute_data(config_path)
        
        if distances is None or similarities is None:
            # Ошибка при загрузке
            ax.text(0.5, 0.5, f'Error loading {dataset_name}',
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=36,
                    color='red')
            all_results.append(None)
            continue
        
        # Сохраняем результаты
        all_results.append((distances, similarities, spearman_corr, spearman_p))
        
        # === СОХРАНЕНИЕ ИНДИВИДУАЛЬНОГО ГРАФИКА ===
        individual_output = f"embedding_correlation_{dataset_name.lower().replace('-', '_')}.pdf"
        create_individual_plot(distances, similarities, spearman_corr, dataset_name, individual_output)
        
        # === ФИКС ВЫБРОСОВ: Определяем границы без выбросов ===
        y_lower = np.percentile(similarities, 1)
        y_upper = np.percentile(similarities, 99)
        
        # Добавляем небольшой отступ к границам
        y_padding = (y_upper - y_lower) * 0.05
        y_min_plot = y_lower - y_padding
        y_max_plot = y_upper + y_padding
        
        # Фильтруем данные для отображения (только для визуализации)
        mask = (similarities >= y_lower) & (similarities <= y_upper)
        distances_plot = distances[mask]
        similarities_plot = similarities[mask]
        
        print(f"  Отображаем {len(distances_plot)} из {len(distances)} точек (убраны выбросы)")
        
        # Scatter plot (только не-выбросы для лучшей визуализации)
        scatter = ax.scatter(
            distances_plot,
            similarities_plot,
            alpha=0.6,
            color='blue',
            s=30,
            edgecolors='white',
            linewidth=0.5,
            label=dataset_name
        )
        
        # Линейная аппроксимация по ВСЕМ данным (для корректной статистики)
        if len(distances) > 1:
            coeffs = np.polyfit(distances, similarities, 1)
            poly_func = np.poly1d(coeffs)
            
            # Строим линию в пределах отображаемых данных
            x_line = np.linspace(distances_plot.min(), distances_plot.max(), 100)
            y_line = poly_func(x_line)
            
            # Рисование линии аппроксимации
            ax.plot(
                x_line,
                y_line,
                color='red',
                linewidth=2.5,
                linestyle='-',
                alpha=0.8,
                label=f'Linear fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}'
            )
        
        # Настройки осей
        ax.set_ylabel('Fraction of matching\npredictions', fontsize=36)
        if idx == 2:  # Только для нижнего графика
            ax.set_xlabel('Euclidean distance\nbetween embeddings', fontsize=36)
        
        # === ФИКС: Устанавливаем фиксированные границы по Y без выбросов ===
        ax.set_ylim([y_min_plot, y_max_plot])
        
        # Автоматически устанавливаем границы X с небольшими отступами
        x_min, x_max = distances_plot.min(), distances_plot.max()
        x_padding = (x_max - x_min) * 0.05
        ax.set_xlim([x_min - x_padding, x_max + x_padding])
        
        # Добавляем название датасета в угол графика
        ax.text(0.02, 0.98, dataset_name, 
                transform=ax.transAxes, 
                fontsize=41,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Вычисляем и добавляем коэффициент корреляции
        if spearman_corr is not None:
            ax.text(0.98, 0.02, f'Spearman ρ = {spearman_corr:.3f}', 
                    transform=ax.transAxes,
                fontsize=32,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
        
        # Сетка
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='gray')
        ax.set_facecolor('white')
        
        print(f"  График для {dataset_name} готов.")
    
    # Общие настройки компоновки
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Сохраняем общий график
    output_path = "embedding_correlation_grid_3x1.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n{'='*60}")
    print(f"Общий график сохранен как: {output_path}")
    
    # Выводим сводную статистику
    print("\nСВОДНАЯ СТАТИСТИКА:")
    print("="*60)
    
    for idx, dataset_name in enumerate(dataset_names):
        if idx < len(all_results) and all_results[idx] is not None:
            distances, similarities, spearman_corr, spearman_p = all_results[idx]
            print(f"\n{dataset_name}:")
            print(f"  Количество пар: {len(distances)}")
            print(f"  Среднее расстояние: {np.mean(distances):.4f} ± {np.std(distances):.4f}")
            print(f"  Средняя схожесть: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
            print(f"  Корреляция Спирмена: {spearman_corr:.4f} (p-value: {spearman_p:.2e})")
        else:
            print(f"\n{dataset_name}: Ошибка загрузки данных")
    
    # Закрываем фигуру
    plt.close()