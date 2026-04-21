import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re
from scipy.stats import spearmanr, kendalltau, sem, t
import sys
import matplotlib
import torch

# Добавляем путь к проекту для импорта модулей
sys.path.insert(1, "../")

# Устанавливаем неинтерактивный бэкенд для избежания проблем с Qt
matplotlib.use("Agg")

# Устанавливаем стиль для публикацию
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)

try:
    from dependencies.data_generator import load_dataset, load_dataset_on_inference
    from dependencies.train_config import TrainConfig
    from train_surrogate import SurrogateTrainer
    from inference_surrogate import InferSurrogate
    from dependencies.GCN import GAT_ver_2
    from tqdm.auto import tqdm
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете скрипт из правильной директории")
    sys.exit(1)

CONFIG_PATH = "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/configs/surrogate_hp_CIFAR100.json"


def get_random_pairs(indices1, indices2, n_pairs):
    """Создает список уникальных пар индексов (i, j), где i != j"""
    pairs = set()
    while len(pairs) < n_pairs:
        i = np.random.choice(indices1)
        j = np.random.choice(indices2)
        if i != j:
            pairs.add(tuple(sorted((i, j))))
    return list(pairs)


def load_single_model(model_path, config):
    """Загружает diversity модель из checkpoint."""
    model = GAT_ver_2(
        config.input_dim,
        config.div_output_dim,
        dropout=config.div_dropout,
        heads=config.div_n_heads,
        output_activation="l2",
        pre_norm=True,
    ).to(config.device)

    state_dict = torch.load(model_path, map_location=config.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_embeddings_for_model(
    model_diversity, config, initial_archs, model_accuracy=None
):
    """Вычисляет эмбеддинги для одной модели diversity."""
    inference = InferSurrogate(config)
    inference.model_diversity = model_diversity

    # Используем переданную accuracy модель или загружаем новую
    if model_accuracy is not None:
        inference.model_accuracy = model_accuracy
    else:
        inference.model_accuracy = GAT_ver_2(
            config.input_dim,
            output_dim=1,
            dropout=config.acc_dropout,
            heads=config.acc_n_heads,
            output_activation="none",
            pre_norm=True,
            pooling="attn",
        ).to(config.device)

        state_dict = torch.load(
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/model_accuracy.pth",
            map_location=config.device,
            weights_only=True,
        )
        inference.model_accuracy.load_state_dict(state_dict)
        inference.model_accuracy.eval()

    archs, accs_np, embs_np = inference._get_embeddings(initial_archs)
    return archs, accs_np, embs_np


def compute_correlations_for_model(
    embs_np, diversity_matrix, test_indices, n_pairs=1000, random_seed=42
):
    """
    Вычисляет корреляции для заданных эмбеддингов и фиксированной матрицы разнообразия.
    """
    # Создаем тест-тест пары (фиксированные для всех моделей)
    max_test_pairs = len(test_indices) * (len(test_indices) - 1) // 2
    actual_n_pairs = min(n_pairs, max_test_pairs)

    # Фиксируем random seed для воспроизводимости пар
    np.random.seed(random_seed)
    test_test_pairs = get_random_pairs(test_indices, test_indices, actual_n_pairs)
    np.random.seed(None)  # Сбрасываем seed

    # Вычисляем метрики для тест-тест пар
    similarities = []
    distances = []
    for i, j in test_test_pairs:
        similarities.append(diversity_matrix[i, j])
        distances.append(np.linalg.norm(embs_np[i] - embs_np[j]))

    similarities = np.array(similarities)
    distances = np.array(distances)

    spearman_corr, spearman_p = spearmanr(distances, similarities)
    kendall_corr, kendall_p = kendalltau(distances, similarities)

    return distances, similarities, spearman_corr, spearman_p, kendall_corr, kendall_p


def find_model_files(models_dir):
    """
    Ищет все .pth файлы в директории.
    """
    models_dir = Path(models_dir)
    pth_files = list(models_dir.rglob("*.pth"))

    # Фильтруем model_accuracy.pth если он есть в путях
    pth_files = [p for p in pth_files if "model_accuracy" not in p.name]

    return pth_files


def load_common_data(config_path, n_pairs=1000, random_seed=42):
    """
    Загружает общие данные, которые используются для всех моделей:
    - конфиг, архитектуры, матрица разнообразия, тестовые индексы
    """
    try:
        print(f"Загрузка конфига: {config_path}")
        params = json.loads(Path(config_path).read_text())
        config = TrainConfig(**params)

        print("  Загрузка датасета...")
        load_dataset(config)

        config.models_dict_path = []
        config.dataset_path = Path(config.dataset_path)
        for file_path in config.dataset_path.rglob("*.json"):
            config.models_dict_path.append(file_path)

        print("  Загрузка архитектур...")
        initial_archs = []
        for arch_json_path in tqdm(
            config.models_dict_path, desc="  Loading pretrained archs"
        ):
            arch = json.loads(arch_json_path.read_text(encoding="utf-8"))
            for key in (
                "test_predictions",
                "test_accuracy",
                "valid_predictions",
                "valid_accuracy",
            ):
                arch.pop(key, None)
            arch["id"] = int(re.search(r"model_(\d+)", str(arch_json_path)).group(1))
            initial_archs.append(arch)

        n_archs = len(initial_archs)
        print(f"  Всего архитектур: {n_archs}")

        # Вычисляем матрицу разнообразия ОДИН РАЗ
        print("\n  Вычисление матрицы разнообразия...")
        trainer = SurrogateTrainer(config)
        trainer.get_diversity_matrix()
        diversity_matrix = trainer.config.diversity_matrix
        print(f"  Размер матрицы разнообразия: {diversity_matrix.shape}")

        # Разделяем индексы на train и test ОДИН РАЗ
        indices = np.arange(n_archs)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.seed(None)

        train_size = int(0.8 * n_archs)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        print(f"  Размер тестового набора: {len(test_indices)} архитектур")

        # Создаем тест-тест пары ОДИН РАЗ
        print("\n  Создание тест-тест пар...")
        test_test_pairs = get_random_pairs(test_indices, test_indices, n_pairs)
        print(f"  Создано {len(test_test_pairs)} пар")

        # Загружаем accuracy модель ОДИН РАЗ
        print("\n  Загрузка accuracy модели...")
        model_accuracy = GAT_ver_2(
            config.input_dim,
            output_dim=1,
            dropout=config.acc_dropout,
            heads=config.acc_n_heads,
            output_activation="none",
            pre_norm=True,
            pooling="attn",
        ).to(config.device)

        state_dict = torch.load(
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/model_accuracy.pth",
            map_location=config.device,
            weights_only=True,
        )
        model_accuracy.load_state_dict(state_dict)
        model_accuracy.eval()

        common_data = {
            "config": config,
            "initial_archs": initial_archs,
            "diversity_matrix": diversity_matrix,
            "test_indices": test_indices,
            "test_test_pairs": test_test_pairs,
            "model_accuracy": model_accuracy,
            "n_archs": n_archs,
        }

        return common_data

    except Exception as e:
        print(f"  Критическая ошибка при загрузке общих данных: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_single_dataset_with_common_data(
    common_data, models_dir, model_index_offset=0
):
    """
    Анализирует одну директорию с моделями, используя общие данные.
    Возвращает список результатов для всех моделей в директории.
    """
    try:
        config = common_data["config"]
        initial_archs = common_data["initial_archs"]
        diversity_matrix = common_data["diversity_matrix"]
        test_indices = common_data["test_indices"]
        test_test_pairs = common_data["test_test_pairs"]
        model_accuracy = common_data["model_accuracy"]

        print(f"\nПоиск моделей в: {models_dir}")
        model_files = find_model_files(models_dir)

        if not model_files:
            print("Не найдено ни одной модели!")
            return []

        print(f"Найдено {len(model_files)} моделей")

        results = []

        for idx, model_path in enumerate(tqdm(model_files, desc="Processing models")):
            try:
                # Загружаем модель diversity
                model_diversity = load_single_model(model_path, config)

                # Вычисляем эмбеддинги для этой модели
                archs, accs_np, embs_np = compute_embeddings_for_model(
                    model_diversity, config, initial_archs, model_accuracy
                )

                # Вычисляем корреляции
                similarities = []
                distances = []
                for i, j in test_test_pairs:
                    similarities.append(diversity_matrix[i, j])
                    distances.append(np.linalg.norm(embs_np[i] - embs_np[j]))

                similarities = np.array(similarities)
                distances = np.array(distances)

                spearman_corr, spearman_p = spearmanr(distances, similarities)
                kendall_corr, kendall_p = kendalltau(distances, similarities)

                results.append(
                    {
                        "model_index": model_index_offset + idx,
                        "model_path": str(model_path),
                        "spearman_corr": spearman_corr,
                        "spearman_p": spearman_p,
                        "kendall_corr": kendall_corr,
                        "kendall_p": kendall_p,
                        "distances": distances,
                        "similarities": similarities,
                    }
                )

            except Exception as e:
                print(f"    Ошибка при обработке модели {model_path}: {e}")
                continue

        print(f"Успешно обработано моделей: {len(results)}")
        return results

    except Exception as e:
        print(f"  Ошибка при анализе директории {models_dir}: {e}")
        import traceback

        traceback.print_exc()
        return []


def compute_mean_and_variance(data):
    """Вычисляет среднее и дисперсию (стандартное отклонение)."""
    if len(data) < 1:
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    std = np.std(data)
    var = np.var(data)

    return mean, std, var


def plot_correlation_vs_samples(datasets_results, output_dir="correlation_analysis"):
    """
    Строит отдельные графики для коэффициентов Спирмена и Кендалла.
    Использует fill_between для отображения стандартного отклонения.
    Все размеры датасетов явно отображаются на оси X.

    datasets_results: список кортежей (models_dir, train_size, results)
    """
    # Извлекаем данные
    train_sizes = []
    spearman_means = []
    spearman_stds = []
    kendall_means = []
    kendall_stds = []

    for models_dir, train_size, results in datasets_results:
        if not results:
            continue

        # Извлекаем коэффициенты корреляции
        spearman_vals = [r["spearman_corr"] for r in results]
        kendall_vals = [r["kendall_corr"] for r in results]

        # Вычисляем статистики
        spearman_mean, spearman_std, spearman_var = compute_mean_and_variance(
            spearman_vals
        )
        kendall_mean, kendall_std, kendall_var = compute_mean_and_variance(kendall_vals)

        train_sizes.append(train_size)
        spearman_means.append(spearman_mean)
        spearman_stds.append(spearman_std)
        kendall_means.append(kendall_mean)
        kendall_stds.append(kendall_std)

        print(f"Размер выборки {train_size}:")
        print(
            f"  Spearman: {spearman_mean:.4f} ± {spearman_std:.4f} (var={spearman_var:.6f})"
        )
        print(
            f"  Kendall: {kendall_mean:.4f} ± {kendall_std:.4f} (var={kendall_var:.6f})"
        )

    # Сортируем по размеру выборки
    sorted_indices = np.argsort(train_sizes)
    train_sizes = np.array(train_sizes)[sorted_indices]
    spearman_means = np.array(spearman_means)[sorted_indices]
    spearman_stds = np.array(spearman_stds)[sorted_indices]
    kendall_means = np.array(kendall_means)[sorted_indices]
    kendall_stds = np.array(kendall_stds)[sorted_indices]

    # Настройки шрифтов для научной публикации
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    # СОЗДАЕМ ОТДЕЛЬНЫЙ ГРАФИК ДЛЯ СПИРМЕНА
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))

    ax1.plot(
        train_sizes,
        (-1) * spearman_means,
        "o-",
        color="#2E86AB",
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor="#2E86AB",
    )

    ax1.fill_between(
        train_sizes,
        (-1) * spearman_means - spearman_stds,
        spearman_means + spearman_stds,
        alpha=0.2,
        color="#2E86AB",
        linewidth=0,
    )

    ax1.set_xlabel("Training set size", fontsize=12)
    ax1.set_ylabel(r"$\rho$", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: явно устанавливаем тики на всех значениях train_sizes
    ax1.set_xticks(train_sizes)
    ax1.set_xticklabels([str(x) for x in train_sizes])

    # Устанавливаем пределы с небольшим отступом
    margin = (train_sizes[-1] - train_sizes[0]) * 0.1
    ax1.set_xlim(train_sizes[0] - margin, train_sizes[-1] + margin)
    ax1.set_ylim(-1, 0)

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_path_spearman = output_dir / "spearman_vs_samples.pdf"
    fig1.savefig(output_path_spearman, dpi=300, bbox_inches="tight")
    print(f"\nГрафик Спирмена сохранен: {output_path_spearman}")
    plt.show()
    plt.close(fig1)

    # СОЗДАЕМ ОТДЕЛЬНЫЙ ГРАФИК ДЛЯ КЕНДАЛЛА
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))

    ax2.plot(
        train_sizes,
        kendall_means,
        "s-",
        color="#A23B72",
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor="#A23B72",
    )

    ax2.fill_between(
        train_sizes,
        kendall_means - kendall_stds,
        kendall_means + kendall_stds,
        alpha=0.2,
        color="#A23B72",
        linewidth=0,
    )

    ax2.set_xlabel("Training set size", fontsize=12)
    ax2.set_ylabel(r"$\tau$", fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: явно устанавливаем тики на всех значениях train_sizes
    ax2.set_xticks(train_sizes)
    ax2.set_xticklabels([str(x) for x in train_sizes])

    # Устанавливаем пределы с небольшим отступом
    ax2.set_xlim(train_sizes[0] - margin, train_sizes[-1] + margin)
    ax2.set_ylim(-1, 0)

    plt.tight_layout()

    output_path_kendall = output_dir / "kendall_vs_samples.pdf"
    fig2.savefig(output_path_kendall, dpi=300, bbox_inches="tight")
    print(f"График Кендалла сохранен: {output_path_kendall}")
    plt.show()
    plt.close(fig2)

    # Создаем сводную таблицу
    summary = []
    for i, (train_size, sp_mean, sp_std, k_mean, k_std) in enumerate(
        zip(train_sizes, spearman_means, spearman_stds, kendall_means, kendall_stds)
    ):
        summary.append(
            {
                "train_size": int(train_size),
                "spearman_mean": float(sp_mean),
                "spearman_std": float(sp_std),
                "spearman_var": float(sp_std**2),
                "kendall_mean": float(k_mean),
                "kendall_std": float(k_std),
                "kendall_var": float(k_std**2),
            }
        )

    with open(output_dir / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Печатаем таблицу
    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Train size':>12} {'Spearman (ρ)':>20} {'Kendall (τ)':>20}")
    print(f"{'':>12} {'mean ± std':>20} {'mean ± std':>20}")
    print("-" * 80)

    for s in summary:
        spearman_str = f"{s['spearman_mean']:.4f} ± {s['spearman_std']:.4f}"
        kendall_str = f"{s['kendall_mean']:.4f} ± {s['kendall_std']:.4f}"
        print(f"{s['train_size']:>12} {spearman_str:>20} {kendall_str:>20}")

    print("=" * 80)

    return fig1, fig2


def create_scatter_subplots_for_each_dataset(datasets_results, max_points_per_plot=500):
    """
    Создает отдельные subplot-ы для каждого набора данных с реальными точками.
    """
    for models_dir, train_size, results in datasets_results:
        if not results:
            continue

        n_models = len(results)
        n_cols = min(5, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Если только один subplot
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(n_rows, n_cols)

        # Строим scatter plot для каждой модели
        for idx, res in enumerate(results):
            if idx >= n_rows * n_cols:
                break

            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Выбираем подмножество точек для визуализации (чтобы не перегружать график)
            if len(res["distances"]) > max_points_per_plot:
                indices = np.random.choice(
                    len(res["distances"]), max_points_per_plot, replace=False
                )
                distances_plot = res["distances"][indices]
                similarities_plot = res["similarities"][indices]
            else:
                distances_plot = res["distances"]
                similarities_plot = res["similarities"]

            # Строим scatter plot
            scatter = ax.scatter(
                distances_plot,
                similarities_plot,
                alpha=0.5,
                s=10,
                c="steelblue",
                edgecolors="none",
            )

            # Добавляем линию тренда
            if len(distances_plot) > 1:
                try:
                    # Линейная регрессия
                    z = np.polyfit(distances_plot, similarities_plot, 1)
                    p = np.poly1d(z)

                    # Генерируем точки для линии
                    x_line = np.linspace(
                        distances_plot.min(), distances_plot.max(), 100
                    )
                    y_line = p(x_line)

                    ax.plot(
                        x_line, y_line, "r-", alpha=0.8, linewidth=1.5, label="Trend"
                    )
                except:
                    pass  # Если не удалось вычислить линию тренда

            # Настройки графика
            ax.set_xlabel("Расстояние", fontsize=9)
            ax.set_ylabel("Схожесть", fontsize=9)

            # Короткое имя модели
            model_name = Path(res["model_path"]).stem
            if len(model_name) > 20:
                model_name = model_name[:17] + "..."

            ax.set_title(
                f"{model_name}\nSpearman: {res['spearman_corr']:.3f}",
                fontsize=10,
                pad=5,
            )

            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # Добавляем коэффициент корреляции в текстовом виде
            text_str = f"ρ = {res['spearman_corr']:.3f}\nτ = {res['kendall_corr']:.3f}"
            ax.text(
                0.05,
                0.95,
                text_str,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Скрываем пустые subplot-ы
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.suptitle(
            f"Директория: {Path(models_dir).name}\nРазмер выборки: {train_size}",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()

        # Сохраняем
        output_name = f"scatter_plots_train_size_{train_size}"
        output_path = f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Сохранен график: {output_path}")
        plt.show()

        # Также сохраняем как PDF
        pdf_path = f"{output_name}.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Сохранен PDF: {pdf_path}")


if __name__ == "__main__":
    # Список кортежей: (путь_к_директории, размер_обучающей_выборки)
    DATASETS = [
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_0",
            0,
        ),
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_250",
            250,
        ),
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_500",
            500,
        ),
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_1000",
            1000,
        ),
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_1500",
            1500,
        ),
        (
            "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/results/surrogates/CIFAR100/ablations/train_size_2000",
            2000,
        ),
    ]

    # Загружаем общие данные ОДИН РАЗ
    print("=" * 60)
    print("ЗАГРУЗКА ОБЩИХ ДАННЫХ")
    print("=" * 60)
    common_data = load_common_data(CONFIG_PATH, n_pairs=1000, random_seed=42)

    if common_data is None:
        print("Не удалось загрузить общие данные!")
        sys.exit(1)

    # Анализируем все наборы данных
    datasets_results = []
    model_index_offset = 0

    for models_dir, train_size in DATASETS:
        print(f"\n{'='*60}")
        print(f"Анализ директории: {models_dir}")
        print(f"Размер обучающей выборки: {train_size}")
        print("=" * 60)

        results = analyze_single_dataset_with_common_data(
            common_data, models_dir, model_index_offset
        )
        datasets_results.append((models_dir, train_size, results))

        if results:
            model_index_offset += len(results)

    # Строим графики зависимости от размера выборки с дисперсией
    plot_correlation_vs_samples(datasets_results)

    # Создаем отдельные subplot-ы с реальными точками для каждого набора данных
    # create_scatter_subplots_for_each_dataset(datasets_results)

    # Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА ПО ВСЕМ ДАННЫМ")
    print("=" * 80)

    all_spearman = []
    all_kendall = []

    for models_dir, train_size, results in datasets_results:
        if results:
            spearman_vals = [r["spearman_corr"] for r in results]
            kendall_vals = [r["kendall_corr"] for r in results]

            all_spearman.extend(spearman_vals)
            all_kendall.extend(kendall_vals)

            print(f"\nРазмер выборки {train_size} ({len(results)} моделей):")
            print(
                f"  Spearman: {np.mean(spearman_vals):.4f} ± {np.std(spearman_vals):.4f}"
            )
            print(
                f"  Kendall: {np.mean(kendall_vals):.4f} ± {np.std(kendall_vals):.4f}"
            )

    if all_spearman:
        print(f"\nВсего моделей: {len(all_spearman)}")
        print(
            f"Общее среднее Spearman: {np.mean(all_spearman):.4f} ± {np.std(all_spearman):.4f}"
        )
        print(
            f"Общее среднее Kendall: {np.mean(all_kendall):.4f} ± {np.std(all_kendall):.4f}"
        )

    print("=" * 80)
