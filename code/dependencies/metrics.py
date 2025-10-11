from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools


def collect_ensemble_stats(
    models: List[torch.nn.Module],
    device: torch.device,
    test_loader: DataLoader,
    n_ece_bins: int,
    developer_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Собирает статистику для ансамбля: точность, ECE, сумма NLL, Oracle NLL, Predictive Disagreement.
    """
    valid_models = [m for m in models if m is not None]
    if not valid_models:
        print("No valid models for ensemble evaluation.")
        return None

    for model in valid_models:
        model.to(device).eval()

    total = 0
    correct_ensemble = 0
    correct_models = [0] * len(valid_models)
    sum_nll = 0.0
    sum_oracle_nll = 0.0

    sum_predictive_disagreement = 0.0  # Сумма несогласия по всему датасету
    num_pred_dis_samples = 0           # Количество обработанных примеров

    n_bins = n_ece_bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_conf_sums = torch.zeros(n_bins)
    bin_acc_sums = torch.zeros(n_bins)
    bin_counts = torch.zeros(n_bins)

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Evaluating ensemble"):
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total += batch_size

            # Для ансамбля (accuracy, NLL)
            avg_output = None
            # Для оракульного лосса
            all_model_probs = []
            # Для predictive disagreement
            all_outputs = []

            for idx, model in enumerate(valid_models):
                output = model(images).softmax(dim=1)  # (batch_size, num_classes)
                all_outputs.append(output.unsqueeze(1))  # [batch, 1, num_classes]
                _, preds = output.max(1)
                correct_models[idx] += (preds == labels).sum().item()

                if avg_output is None:
                    avg_output = torch.zeros_like(output)
                avg_output += output

                # Для oracle loss: вероятности правильного класса
                p_i = output[torch.arange(batch_size), labels]
                all_model_probs.append(p_i.unsqueeze(1))  # делаем столбец

            # Ансамбль вероятностей
            avg_output /= len(valid_models)
            confidences, preds_ens = avg_output.max(1)
            correct_ens_batch = (preds_ens == labels).float()
            correct_ensemble += correct_ens_batch.sum().item()

            # NLL для ансамбля
            eps = 1e-12
            p_targets = avg_output[torch.arange(batch_size), labels]
            nll_batch = -torch.log(p_targets + eps)
            sum_nll += nll_batch.sum().item()

            # Oracle loss: минимум NLL по всем моделям на каждом объекте батча
            all_model_probs = torch.cat(all_model_probs, dim=1)  # (batch_size, num_models)
            oracle_nll_batch = -torch.log(all_model_probs + eps)  # (batch_size, num_models)
            min_oracle_nll_per_sample, _ = oracle_nll_batch.min(dim=1)  # (batch_size, )
            sum_oracle_nll += min_oracle_nll_per_sample.sum().item()

            # Predictive disagreement (L1-норма) --------------------
            # Формируем (batch_size, num_models, num_classes)
            all_outputs_tensor = torch.cat(all_outputs, dim=1)  # (batch_size, num_models, num_classes)
            num_models = len(valid_models)

            # Для каждой пары (i, j) считаем l1 расстояние между предсказаниями на каждом объекте
            disagreements = []
            model_indices = list(range(num_models))
            for i, j in itertools.combinations(model_indices, 2):
                # L1 по последней оси (num_classes)
                l1 = (all_outputs_tensor[:, i, :] - all_outputs_tensor[:, j, :]).abs().sum(dim=1)
                disagreements.append(l1)
            # (num_pairs, batch_size) -> (batch_size, )
            if disagreements:
                disagreements_tensor = torch.stack(disagreements, dim=1)  # (batch_size, num_pairs)
                batch_pred_dis = disagreements_tensor.mean(dim=1)  # (batch_size,)
                sum_predictive_disagreement += batch_pred_dis.sum().item()
                num_pred_dis_samples += batch_size

            # ECE
            confidences = confidences.cpu().float()
            correct_ens_batch = correct_ens_batch.cpu()
            for conf, correct in zip(confidences, correct_ens_batch):
                bin_idx = torch.bucketize(conf, bin_boundaries, right=True) - 1
                bin_idx = bin_idx.clamp(min=0, max=n_bins - 1)
                bin_counts[bin_idx] += 1
                bin_conf_sums[bin_idx] += conf
                bin_acc_sums[bin_idx] += correct

            if developer_mode:
                break

    # Среднее несогласие на примере:
    predictive_disagreement = sum_predictive_disagreement / num_pred_dis_samples if num_pred_dis_samples > 0 else float('nan')
    # Средняя ошибка моделей
    avg_model_err = 1.0 - sum(correct_models) / (len(correct_models) * total) if total > 0 else float("nan")
    # Нормализованное несогласие
    normalized_predictive_disagreement = predictive_disagreement / avg_model_err if avg_model_err > 0 else float("nan")

    return {
        "total": total,
        "correct_ensemble": correct_ensemble,
        "correct_models": correct_models,
        "bin_counts": bin_counts,
        "bin_conf_sums": bin_conf_sums,
        "bin_acc_sums": bin_acc_sums,
        "n_bins": n_bins,
        "num_models": len(valid_models),
        "sum_nll": sum_nll,
        "sum_oracle_nll": sum_oracle_nll,
        "predictive_disagreement": predictive_disagreement,
        "normalized_predictive_disagreement": normalized_predictive_disagreement,
        "avg_model_err": avg_model_err,
    }


def calculate_ece(stats: dict) -> float:
    """
    Вычисляет Expected Calibration Error (ECE) по словарю stats,
    содержащему ключи: bin_counts, bin_conf_sums, bin_acc_sums, n_bins, total.
    """
    bin_counts = stats["bin_counts"]
    bin_conf_sums = stats["bin_conf_sums"]
    bin_acc_sums = stats["bin_acc_sums"]
    n_bins = stats["n_bins"]
    total = stats["total"]

    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_conf_avg = bin_conf_sums[i] / bin_counts[i]
            bin_acc_avg = bin_acc_sums[i] / bin_counts[i]
            bin_weight = bin_counts[i] / total
            ece += bin_weight * abs(bin_conf_avg - bin_acc_avg)
    return ece


def calculate_nll(stats: dict) -> float:
    """
    Вычисляет Negative Log-Likelihood (NLL) по словарю stats,
    содержащему ключи: 'sum_nll' и 'total'.
    """
    sum_nll = stats["sum_nll"]
    total = stats["total"]
    return sum_nll / total if total > 0 else float("nan")


def calculate_oracle_nll(stats):
    sum_oracle_nll = stats["sum_oracle_nll"]
    total = stats["total"]

    return sum_oracle_nll / total if total > 0 else float("nan")
