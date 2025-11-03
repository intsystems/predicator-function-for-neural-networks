import glob
import re
import numpy as np
import os

DATA_DIR = "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/output"
# DATA_DIR = '.' если текущая

METRICS = {
    'Ensemble Top-1 Accuracy':         r'Ensemble Top-1 Accuracy:\s+([\d.]+)%',
    'Ensemble ECE':                    r'Ensemble ECE:\s+([\d.]+)',
    'Ensemble NLL':                    r'Ensemble NLL:\s+([\d.]+)',
    'Oracle Ensemble NLL':             r'Oracle Ensemble NLL:\s+([\d.]+)',
    'Normalized Predictive Disagreement': r'Normalized Predictive Disagreement:\s+([\d.]+)',
    'Number of models':                r'Number of models:\s+(\d+)',
}

# Паттерны для разных типов атак
ATTACK_SECTION_PATTERN = re.compile(
    r'--- Adversarial Attack Results \((\w+)\) ---\n(.*?)(?=--- Adversarial Attack Results|\Z)',
    re.DOTALL
)

EPS_PATTERN = re.compile(
    r"Epsilon = ([\d.]+)\n\s+Ensemble accuracy:\s+([\d.]+)%"
)

def extract_metrics_from_file(filename):
    with open(filename, encoding="utf-8") as f:
        text = f.read()

    metrics = {}
    
    # Извлекаем основные метрики
    for k, pat in METRICS.items():
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            metrics[k] = val
        else:
            metrics[k] = np.nan

    # Извлекаем результаты атак
    for attack_match in ATTACK_SECTION_PATTERN.finditer(text):
        attack_type = attack_match.group(1)  # FGSM, BIM, PGD
        attack_section = attack_match.group(2)
        
        # Извлекаем результаты для каждого epsilon
        for eps_match in EPS_PATTERN.finditer(attack_section):
            eps = float(eps_match.group(1))
            acc = float(eps_match.group(2))
            
            # Формируем название метрики
            metric_name = f'Ensemble {attack_type} acc @ {eps:.4f}'
            metrics[metric_name] = acc

    return metrics

# Собираем все файлы
files_pattern = os.path.join(DATA_DIR, "ensemble_results_*_*.txt")
files = sorted(glob.glob(files_pattern))

print(f"Найдено файлов: {len(files)}")

all_metrics = []
for fname in files:
    m = extract_metrics_from_file(fname)
    all_metrics.append(m)

from collections import defaultdict
values_by_metric = defaultdict(list)
for metdict in all_metrics:
    for k, v in metdict.items():
        values_by_metric[k].append(v)

# Группируем метрики по категориям
basic_metrics = []
fgsm_metrics = []
bim_metrics = []
pgd_metrics = []

for metric in sorted(values_by_metric):
    if 'FGSM' in metric:
        fgsm_metrics.append(metric)
    elif 'BIM' in metric:
        bim_metrics.append(metric)
    elif 'PGD' in metric:
        pgd_metrics.append(metric)
    else:
        basic_metrics.append(metric)

# Функция для сортировки метрик атак по epsilon
def sort_attack_metrics(metrics_list):
    def get_epsilon(metric_name):
        match = re.search(r'@ ([\d.]+)', metric_name)
        return float(match.group(1)) if match else 0
    return sorted(metrics_list, key=get_epsilon)

fgsm_metrics = sort_attack_metrics(fgsm_metrics)
bim_metrics = sort_attack_metrics(bim_metrics)
pgd_metrics = sort_attack_metrics(pgd_metrics)

# Формируем таблицу
rows = []
rows.append("# Результаты оценки ансамбля\n")
rows.append("## Основные метрики\n")
rows.append("| Метрика | Среднее ± Ст.откл. |")
rows.append("|---|---|")

for metric in basic_metrics:
    vals = np.array(values_by_metric[metric], dtype=float)
    if np.all(np.isnan(vals)):
        continue
    avg = np.nanmean(vals)
    std = np.nanstd(vals)
    if "Accuracy" in metric:
        avg_str = f"{avg:.2f}% ± {std:.2f}"
    else:
        avg_str = f"{avg:.4f} ± {std:.4f}"
    rows.append(f"| {metric} | {avg_str} |")

# FGSM атаки
if fgsm_metrics:
    rows.append("\n## FGSM атаки\n")
    rows.append("| Epsilon | Accuracy (среднее ± ст.откл.) |")
    rows.append("|---|---|")
    
    for metric in fgsm_metrics:
        vals = np.array(values_by_metric[metric], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        eps_match = re.search(r'@ ([\d.]+)', metric)
        eps = eps_match.group(1) if eps_match else "?"
        rows.append(f"| {eps} | {avg:.2f}% ± {std:.2f} |")

# BIM атаки
if bim_metrics:
    rows.append("\n## BIM атаки\n")
    rows.append("| Epsilon | Accuracy (среднее ± ст.откл.) |")
    rows.append("|---|---|")
    
    for metric in bim_metrics:
        vals = np.array(values_by_metric[metric], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        eps_match = re.search(r'@ ([\d.]+)', metric)
        eps = eps_match.group(1) if eps_match else "?"
        rows.append(f"| {eps} | {avg:.2f}% ± {std:.2f} |")

# PGD атаки
if pgd_metrics:
    rows.append("\n## PGD атаки\n")
    rows.append("| Epsilon | Accuracy (среднее ± ст.откл.) |")
    rows.append("|---|---|")
    
    for metric in pgd_metrics:
        vals = np.array(values_by_metric[metric], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        avg = np.nanmean(vals)
        std = np.nanstd(vals)
        eps_match = re.search(r'@ ([\d.]+)', metric)
        eps = eps_match.group(1) if eps_match else "?"
        rows.append(f"| {eps} | {avg:.2f}% ± {std:.2f} |")

table_text = "\n".join(rows)
output_file = os.path.join(DATA_DIR, "output_metrics.md")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(table_text)

print(f"Готово! Результаты сохранены в {output_file}")
print(f"Обработано экспериментов: {len(all_metrics)}")
