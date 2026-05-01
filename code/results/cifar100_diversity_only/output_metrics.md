# Результаты оценки ансамбля

## Основные метрики

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Average Model Top-1 Accuracy | 80.08% ± 0.21 |
| Ensemble Top-1 Accuracy | 84.77% ± 0.07 |
| Ensemble improvement over avg model | 4.69% ± 0.18 |
| NLL improvement (Oracle - Ensemble) | -0.29% ± 0.01 |
| Number of models | 5 |
| Total samples | 10000 |

## Метрики калибровки

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Brier Score | 0.2373 ± 0.0015 |
| Expected Calibration Error (ECE) | 0.1373 ± 0.0027 |
| Negative Log-Likelihood (NLL) | 0.6983 ± 0.0020 |
| Oracle NLL | 0.4074 ± 0.0042 |

## Метрики разнообразия

| Метрика | Среднее ± Ст.откл. |
|---|---|
| Ambiguity (Ensemble Benefit) | 0.0469 ± 0.0018 |
| Normalized Pred. Disagreement | 0.2096 ± 0.0042 |
| Predictive Disagreement | 0.4191 ± 0.0082 |