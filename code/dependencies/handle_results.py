import glob
import re
import numpy as np
import os

DATA_DIR = "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/results/fashionmnist_deepens"
# DATA_DIR = '.' если текущая

METRICS = {
    'Ensemble Top-1 Accuracy':         r'Ensemble Top-1 Accuracy:\s+([\d.]+)%',
    'Ensemble ECE':                    r'Ensemble ECE:\s+([\d.]+)',
    'Ensemble NLL':                    r'Ensemble NLL:\s+([\d.]+)',
    'Oracle Ensemble NLL':             r'Oracle Ensemble NLL:\s+([\d.]+)',
    'Normalized Predictive Disagreement': r'Normalized Predictive Disagreement:\s+([\d.]+)',
    'Number of models':                r'Number of models:\s+(\d+)',
}

EPS_PATTERN = re.compile(
    r"Epsilon = ([\d.]+)\n"
    r"\s+Ensemble accuracy:\s+([\d.]+)%"
)

def extract_metrics_from_file(filename):
    with open(filename, encoding="utf-8") as f:
        text = f.read()

    metrics = {}
    for k, pat in METRICS.items():
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            metrics[k] = val
        else:
            metrics[k] = np.nan

    adv_results = {}
    for match in EPS_PATTERN.finditer(text):
        eps = float(match.group(1))
        acc = float(match.group(2))
        adv_results[eps] = acc

    if adv_results:
        metrics_adv = {f'Ensemble FGSM acc @ {eps:.3f}': acc for eps, acc in sorted(adv_results.items())}
        metrics.update(metrics_adv)

    return metrics

# Собираем все файлы
files_pattern = os.path.join(DATA_DIR, "ensemble_results_*_*.txt")
files = sorted(glob.glob(files_pattern))

all_metrics = []
for fname in files:
    m = extract_metrics_from_file(fname)
    all_metrics.append(m)

from collections import defaultdict
values_by_metric = defaultdict(list)
for metdict in all_metrics:
    for k, v in metdict.items():
        values_by_metric[k].append(v)

rows = []
rows.append("| Метрика | Среднее ± Ст.откл. |")
rows.append("|---|---|")

for metric in sorted(values_by_metric):
    vals = np.array(values_by_metric[metric], dtype=float)
    if np.all(np.isnan(vals)):
        continue
    avg = np.nanmean(vals)
    std = np.nanstd(vals)
    if "Accuracy" in metric:
        avg_str = f"{avg:.2f}% ± {std:.4f}"
    else:
        avg_str = f"{avg:.2f} ± {std:.2f}"
    rows.append(f"| {metric} | {avg_str} |")

table_text = "\n".join(rows)
output_file = os.path.join(DATA_DIR, "output_metrics.md")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(table_text)

print(f"Готово! Результаты сохранены в {output_file}")
