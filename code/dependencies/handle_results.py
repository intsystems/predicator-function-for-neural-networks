import glob
import re
import numpy as np
import os
import argparse

def extract_metrics_from_file(filename):
    """Извлекает метрики из файла с результатами."""
    METRICS = {
        'Number of models':                r'Number of models:\s+(\d+)',
        'Total samples':                   r'Total samples:\s+(\d+)',
        'Ensemble Top-1 Accuracy':         r'Ensemble Top-1 Accuracy:\s+([\d.]+)%',
        'Average Model Top-1 Accuracy':    r'Average Model Top-1 Accuracy:\s+([\d.]+)%',
        'Expected Calibration Error (ECE)': r'Expected Calibration Error \(ECE\):\s+([\d.]+)',
        'Negative Log-Likelihood (NLL)':   r'Negative Log-Likelihood \(NLL\):\s+([\d.]+)',
        'Oracle NLL':                      r'Oracle NLL:\s+([\d.]+)',
        'Brier Score':                     r'Brier Score:\s+([\d.]+)',
        'Ambiguity (Ensemble Benefit)':    r'Ambiguity \(Ensemble Benefit\):\s+([\d.]+)',
        'Predictive Disagreement':         r'Predictive Disagreement:\s+([\d.]+)',
        'Normalized Pred. Disagreement':   r'Normalized Pred\. Disagreement:\s+([\d.]+)',
        'Ensemble improvement over avg model': r'Ensemble improvement over avg model:\s+([\d.]+)%',
        'NLL improvement (Oracle - Ensemble)': r'NLL improvement \(Oracle - Ensemble\):\s+([-\d.]+)',
    }

    ATTACK_SECTION_PATTERN = re.compile(
        r'ADVERSARIAL ATTACK RESULTS \((\w+)\)\n=+\n(.*?)(?=\n=+\n|ADDITIONAL INFORMATION|\Z)',
        re.DOTALL
    )

    EPS_PATTERN = re.compile(
        r"Epsilon = ([\d.]+)\n\s+Ensemble accuracy:\s+([\d.]+)%"
    )
    
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
    
    # Извлекаем метрики падения точности для разных атак
    fgsm_drop = re.search(r'FGSM accuracy drop \(eps=([\d.]+)\):\s+([\d.]+)%', text)
    if fgsm_drop:
        eps = float(fgsm_drop.group(1))
        drop = float(fgsm_drop.group(2))
        metrics[f'FGSM accuracy drop @ {eps:.4f}'] = drop
    
    bim_drop = re.search(r'BIM accuracy drop \(eps=([\d.]+)\):\s+([\d.]+)%', text)
    if bim_drop:
        eps = float(bim_drop.group(1))
        drop = float(bim_drop.group(2))
        metrics[f'BIM accuracy drop @ {eps:.4f}'] = drop
    
    pgd_drop = re.search(r'PGD accuracy drop \(eps=([\d.]+)\):\s+([\d.]+)%', text)
    if pgd_drop:
        eps = float(pgd_drop.group(1))
        drop = float(pgd_drop.group(2))
        metrics[f'PGD accuracy drop @ {eps:.4f}'] = drop

    return metrics

def sort_attack_metrics(metrics_list):
    """Сортирует метрики атак по epsilon."""
    def get_epsilon(metric_name):
        match = re.search(r'@ ([\d.]+)', metric_name)
        return float(match.group(1)) if match else 0
    return sorted(metrics_list, key=get_epsilon)

def generate_markdown_table(values_by_metric, data_dir):
    """Генерирует markdown таблицу с результатами."""
    # Группируем метрики по категориям
    basic_metrics = []
    calibration_metrics = []
    diversity_metrics = []
    fgsm_metrics = []
    bim_metrics = []
    pgd_metrics = []
    accuracy_drop_metrics = []

    for metric in sorted(values_by_metric):
        if 'FGSM' in metric and 'drop' not in metric:
            fgsm_metrics.append(metric)
        elif 'BIM' in metric and 'drop' not in metric:
            bim_metrics.append(metric)
        elif 'PGD' in metric and 'drop' not in metric:
            pgd_metrics.append(metric)
        elif 'drop' in metric:
            accuracy_drop_metrics.append(metric)
        elif metric in ['Expected Calibration Error (ECE)', 'Negative Log-Likelihood (NLL)', 
                        'Oracle NLL', 'Brier Score']:
            calibration_metrics.append(metric)
        elif metric in ['Ambiguity (Ensemble Benefit)', 'Predictive Disagreement', 
                        'Normalized Pred. Disagreement']:
            diversity_metrics.append(metric)
        else:
            basic_metrics.append(metric)

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
        if "Accuracy" in metric or "improvement" in metric:
            avg_str = f"{avg:.2f}% ± {std:.2f}"
        elif metric in ['Number of models', 'Total samples']:
            avg_str = f"{int(avg)}"
        else:
            avg_str = f"{avg:.4f} ± {std:.4f}"
        rows.append(f"| {metric} | {avg_str} |")

    # Калибровка
    if calibration_metrics:
        rows.append("\n## Метрики калибровки\n")
        rows.append("| Метрика | Среднее ± Ст.откл. |")
        rows.append("|---|---|")
        
        for metric in calibration_metrics:
            vals = np.array(values_by_metric[metric], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            avg = np.nanmean(vals)
            std = np.nanstd(vals)
            avg_str = f"{avg:.4f} ± {std:.4f}"
            rows.append(f"| {metric} | {avg_str} |")

    # Разнообразие
    if diversity_metrics:
        rows.append("\n## Метрики разнообразия\n")
        rows.append("| Метрика | Среднее ± Ст.откл. |")
        rows.append("|---|---|")
        
        for metric in diversity_metrics:
            vals = np.array(values_by_metric[metric], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            avg = np.nanmean(vals)
            std = np.nanstd(vals)
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

    # Падение точности
    if accuracy_drop_metrics:
        rows.append("\n## Падение точности при атаках\n")
        rows.append("| Тип атаки | Epsilon | Падение точности (среднее ± ст.откл.) |")
        rows.append("|---|---|---|")
        
        for metric in sorted(accuracy_drop_metrics):
            vals = np.array(values_by_metric[metric], dtype=float)
            if np.all(np.isnan(vals)):
                continue
            avg = np.nanmean(vals)
            std = np.nanstd(vals)
            
            attack_type = metric.split()[0]  # FGSM, BIM, или PGD
            eps_match = re.search(r'@ ([\d.]+)', metric)
            eps = eps_match.group(1) if eps_match else "?"
            rows.append(f"| {attack_type} | {eps} | {avg:.2f}% ± {std:.2f} |")

    return "\n".join(rows)

def main():
    parser = argparse.ArgumentParser(
        description='Агрегация метрик из файлов результатов оценки ансамбля'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.',
        help='Директория с файлами ensemble_results_*_*.txt (по умолчанию: текущая директория)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_metrics.md',
        help='Имя выходного файла (по умолчанию: output_metrics.md)'
    )
    
    args = parser.parse_args()
    
    # Проверяем существование директории
    if not os.path.isdir(args.data_dir):
        print(f"Ошибка: директория '{args.data_dir}' не существует")
        return
    
    # Собираем все файлы
    files_pattern = os.path.join(args.data_dir, "ensemble_results_*_*.txt")
    files = sorted(glob.glob(files_pattern))
    
    if len(files) == 0:
        print(f"Ошибка: не найдено файлов по паттерну '{files_pattern}'")
        return
    
    print(f"Найдено файлов: {len(files)}")
    print(f"Директория: {os.path.abspath(args.data_dir)}")
    
    # Извлекаем метрики из всех файлов
    all_metrics = []
    for fname in files:
        try:
            m = extract_metrics_from_file(fname)
            all_metrics.append(m)
        except Exception as e:
            print(f"Предупреждение: ошибка при обработке файла {fname}: {e}")
    
    if len(all_metrics) == 0:
        print("Ошибка: не удалось извлечь метрики ни из одного файла")
        return
    
    # Группируем значения по метрикам
    from collections import defaultdict
    values_by_metric = defaultdict(list)
    for metdict in all_metrics:
        for k, v in metdict.items():
            values_by_metric[k].append(v)
    
    # Генерируем таблицу
    table_text = generate_markdown_table(values_by_metric, args.data_dir)
    
    # Сохраняем результат
    output_file = os.path.join(args.data_dir, args.output)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(table_text)
    
    print(f"Готово! Результаты сохранены в {output_file}")
    print(f"Обработано экспериментов: {len(all_metrics)}")

if __name__ == "__main__":
    main()
