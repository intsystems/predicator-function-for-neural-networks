# Результаты оценки ансамбля

## Основные метрики

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Average Model Top-1 Accuracy | 80.54% ± 0.12 |
| Ensemble Top-1 Accuracy | 85.07% ± 0.10 |
| Ensemble improvement over avg model | 4.53% ± 0.03 |
| NLL improvement (Oracle - Ensemble) | -0.29% ± 0.00 |
| Number of models | 5 |
| Total samples | 10000 |

## Метрики калибровки

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Brier Score | 0.2335 ± 0.0010 |
| Expected Calibration Error (ECE) | 0.1363 ± 0.0015 |
| Negative Log-Likelihood (NLL) | 0.6876 ± 0.0055 |
| Oracle NLL | 0.4006 ± 0.0044 |

## Метрики разнообразия

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Ambiguity (Ensemble Benefit) | 0.0453 ± 0.0003 |
| Normalized Pred. Disagreement | 0.2032 ± 0.0010 |
| Predictive Disagreement | 0.4063 ± 0.0021 |