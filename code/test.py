import json
import re
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
import torch
from nni.nas.space import model_context

from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace

def load_model_from_file(file_path: str, pth_path: str):
    with open(file_path, "r") as f:
        arch = json.load(f)
        arch = arch["architecture"]

    dataset = "cifar100"
    with model_context(arch):
        model = DartsSpace(
            width=4,
            num_cells=3,
            dataset=dataset,
        )

    model.to("cpu")
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    return model


def collect_ensemble_accuracy(folder_path: str, output_file: str = "results.txt"):
    folder = Path(folder_path)
    accuracies = []

    for txt_file in folder.glob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Ensemble Top-1 Accuracy:"):
                    # вытащим число с плавающей точкой
                    match = re.search(r"([\d.]+)%", line)
                    if match:
                        accuracies.append(match.group(1))
                    break

    # сохраним только числа, по одной строке
    output_path = folder / output_file
    with output_path.open("w", encoding="utf-8") as out:
        out.write("\n".join(accuracies))

    print(f"Собрано {len(accuracies)} значений, результат сохранён в {output_path}")

def t_test_from_file(results_file: str, popmean: float = 25.0, alternative: str = "two-sided"):
    """
    Считает t-тест для проверки гипотезы H0: mean == popmean.
    alternative:
        "two-sided" — двусторонняя гипотеза (по умолчанию)
        "less"       — левосторонняя (H1: mean < popmean)
        "greater"    — правосторонняя (H1: mean > popmean)
    """
    values = np.loadtxt(results_file)
    t_stat, p_value_two_sided = stats.ttest_1samp(values, popmean)

    if alternative == "two-sided":
        p_value = p_value_two_sided
    elif alternative == "less":
        p_value = p_value_two_sided / 2 if t_stat < 0 else 1 - p_value_two_sided / 2
    elif alternative == "greater":
        p_value = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")

    print(f"T-statistic = {t_stat:.4f}, p-value = {p_value:.6f} (alternative='{alternative}')")
    return t_stat, p_value


def plot_confidence_interval(results_file: str, confidence: float = 0.95):
    """
    Строит доверительный интервал для среднего и рисует его.
    """
    values = np.loadtxt(results_file)
    mean = np.mean(values)
    sem = stats.sem(values)  # стандартная ошибка среднего
    h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)  # половина ширины интервала

    ci_low, ci_high = mean - h, mean + h

    # Рисуем
    plt.figure(figsize=(6, 4))
    plt.errorbar(1, mean, yerr=h, fmt='o', capsize=5, color="blue")
    plt.axhline(mean, color="blue", linestyle="--", label=f"Mean = {mean:.2f}")
    plt.axhline(ci_low, color="red", linestyle=":", label=f"CI lower = {ci_low:.2f}")
    plt.axhline(ci_high, color="red", linestyle=":", label=f"CI upper = {ci_high:.2f}")
    plt.xlim(0.5, 1.5)
    plt.xticks([])
    plt.ylabel("Accuracy (%)")
    plt.title(f"{int(confidence*100)}% Confidence Interval")
    plt.legend()
    plt.show()

    return ci_low, ci_high

if __name__ == "__main__":
    # model = load_model_from_file(
    #     "datasets/cifar100_archs/model_1000.json",
    #     "datasets/cifar100_pth/model_1000.pth",
    # )

    # collect_ensemble_accuracy("output/")

    t_test_from_file("results/cifar100_random.txt", popmean=30.66, alternative="less")

    plot_confidence_interval("results/cifar100_random.txt", confidence=0.95)
     
    print("END")
