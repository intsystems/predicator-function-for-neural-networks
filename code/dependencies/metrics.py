from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.epochs = []
        self._has_data = False

    def on_train_epoch_end(self, trainer, pl_module):
        """Собираем ТОЛЬКО обучающие метрики и LR"""
        current_epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        epoch_data = {}
        
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'].item())
            epoch_data['train_loss'] = metrics['train_loss'].item()
        if 'train_acc' in metrics:
            self.train_accs.append(metrics['train_acc'].item())
            epoch_data['train_acc'] = metrics['train_acc'].item()
        
        try:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            self.lrs.append(lr)
            epoch_data['lr'] = lr
        except Exception as e:
            if self.lrs:
                self.lrs.append(self.lrs[-1])
            else:
                self.lrs.append(0.0)
        
        self.epochs.append(current_epoch)
        self._has_data = True
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Собираем ТОЛЬКО валидационные метрики"""
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        
        epoch_data = {}
        
        if 'val_loss' in metrics:
            if len(self.val_losses) < len(self.epochs):
                self.val_losses.append(metrics['val_loss'].item())
                epoch_data['val_loss'] = metrics['val_loss'].item()
        if 'val_acc' in metrics:
            if len(self.val_accs) < len(self.epochs):
                self.val_accs.append(metrics['val_acc'].item())
                epoch_data['val_acc'] = metrics['val_acc'].item()
    



def collect_ensemble_stats(
    models: List[torch.nn.Module],
    device: torch.device,
    test_loader: DataLoader,
    n_ece_bins: int,
    developer_mode: bool = False,
    mean=0,
    std=1,
) -> Optional[Dict[str, Any]]:
    """
    Собирает статистику для ансамбля: точность, ECE, NLL, Oracle NLL, 
    Predictive Disagreement, Ambiguity.
    
    Args:
        models: Список моделей ансамбля
        device: Устройство для вычислений
        test_loader: DataLoader с тестовыми данными
        n_ece_bins: Количество бинов для ECE
        developer_mode: Режим разработки (один батч)
        mean: Среднее для денормализации
        std: Стд. отклонение для денормализации
    
    Returns:
        Словарь со статистикой ансамбля
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
    sum_brier_score = 0.0

    sum_predictive_disagreement = 0.0
    num_pred_dis_samples = 0

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

            avg_output = None
            all_model_probs = []
            all_outputs = []

            for idx, model in enumerate(valid_models):
                output = model(images).softmax(dim=1)
                all_outputs.append(output.unsqueeze(1))
                _, preds = output.max(1)
                correct_models[idx] += (preds == labels).sum().item()

                if avg_output is None:
                    avg_output = torch.zeros_like(output)
                avg_output += output

                p_i = output[torch.arange(batch_size, device=device), labels]
                all_model_probs.append(p_i.unsqueeze(1))

            # Ансамбль вероятностей
            avg_output /= len(valid_models)
            confidences, preds_ens = avg_output.max(1)
            correct_ens_batch = (preds_ens == labels).float()
            correct_ensemble += correct_ens_batch.sum().item()

            # NLL для ансамбля
            eps = 1e-12
            p_targets = avg_output[torch.arange(batch_size, device=device), labels]
            nll_batch = -torch.log(p_targets + eps)
            sum_nll += nll_batch.sum().item()

            # Brier Score
            one_hot_labels = F.one_hot(labels, num_classes=avg_output.size(1)).float()
            brier_batch = ((avg_output - one_hot_labels) ** 2).sum(dim=1)
            sum_brier_score += brier_batch.sum().item()

            # Oracle NLL: минимальный NLL среди всех моделей
            all_model_probs = torch.cat(all_model_probs, dim=1)
            oracle_nll_batch = -torch.log(all_model_probs + eps)
            min_oracle_nll_per_sample, _ = oracle_nll_batch.min(dim=1)
            sum_oracle_nll += min_oracle_nll_per_sample.sum().item()

            # Predictive disagreement: среднее L1-расстояние между всеми парами моделей
            all_outputs_tensor = torch.cat(all_outputs, dim=1)
            num_models = len(valid_models)

            disagreements = []
            model_indices = list(range(num_models))
            for i, j in itertools.combinations(model_indices, 2):
                # L1 расстояние между распределениями вероятностей
                l1 = (
                    (all_outputs_tensor[:, i, :] - all_outputs_tensor[:, j, :])
                    .abs()
                    .sum(dim=1)
                )
                disagreements.append(l1)

            if disagreements:
                disagreements_tensor = torch.stack(disagreements, dim=1)
                batch_pred_dis = disagreements_tensor.mean(dim=1)
                sum_predictive_disagreement += batch_pred_dis.sum().item()
                num_pred_dis_samples += batch_size

            # ECE: улучшенный биннинг
            confidences_cpu = confidences.cpu().float()
            correct_ens_batch_cpu = correct_ens_batch.cpu()
            
            for conf, correct in zip(confidences_cpu, correct_ens_batch_cpu):
                # Используем searchsorted для более точного биннинга
                bin_idx = torch.searchsorted(bin_boundaries[1:], conf, right=False)
                bin_idx = bin_idx.clamp(min=0, max=n_bins - 1)
                bin_counts[bin_idx] += 1
                bin_conf_sums[bin_idx] += conf
                bin_acc_sums[bin_idx] += correct

            if developer_mode:
                break

    # Метрики ансамбля
    ensemble_accuracy = correct_ensemble / total if total > 0 else 0.0
    ensemble_error = 1.0 - ensemble_accuracy

    # Средняя точность и ошибка отдельных моделей
    avg_model_accuracy = (
        sum(correct_models) / (len(correct_models) * total)
        if total > 0
        else 0.0
    )
    avg_model_error = 1.0 - avg_model_accuracy

    # Predictive Disagreement
    # L1-расстояние между распределениями вероятностей лежит в [0, 2]
    predictive_disagreement = (
        sum_predictive_disagreement / num_pred_dis_samples
        if num_pred_dis_samples > 0
        else float("nan")
    )
    # Нормализация на [0, 1]
    normalized_predictive_disagreement = predictive_disagreement / 2.0

    # Ambiguity (из Ambiguity Decomposition от Krogh-Vedelsby)
    # Ambiguity = Avg_Model_Error - Ensemble_Error
    # Показывает, насколько разнообразие моделей снижает ошибку ансамбля
    ambiguity = avg_model_error - ensemble_error

    # --- Adversarial атаки ---
    # FGSM
    fgsm_epsilons = [0, 0.02, 0.04, 0.08, 0.1]
    fgsm_results = adversarial_attack(
        valid_models,
        test_loader,
        fgsm_epsilons,
        device=device,
        mean=mean,
        std=std,
        attack_type="FGSM",
        developer_mode=developer_mode,
    )

    # BIM
    bim_epsilons = [0, 0.02, 0.04, 0.08, 0.1]
    bim_results = adversarial_attack(
        valid_models,
        test_loader,
        bim_epsilons,
        device=device,
        mean=mean,
        std=std,
        attack_type="BIM",
        num_steps=10,
        developer_mode=developer_mode,
    )

    # PGD
    pgd_epsilons = [0, 0.02, 0.04, 0.08, 0.1]
    pgd_results = adversarial_attack(
        valid_models,
        test_loader,
        pgd_epsilons,
        device=device,
        mean=mean,
        std=std,
        attack_type="PGD",
        num_steps=10,
        developer_mode=developer_mode,
    )

    return {
        "total": total,
        "correct_ensemble": correct_ensemble,
        "correct_models": correct_models,
        "ensemble_accuracy": ensemble_accuracy,
        "ensemble_error": ensemble_error,
        "avg_model_accuracy": avg_model_accuracy,
        "avg_model_error": avg_model_error,
        "bin_counts": bin_counts,
        "bin_conf_sums": bin_conf_sums,
        "bin_acc_sums": bin_acc_sums,
        "n_bins": n_bins,
        "num_models": len(valid_models),
        "sum_nll": sum_nll,
        "sum_oracle_nll": sum_oracle_nll,
        "sum_brier_score": sum_brier_score,
        "predictive_disagreement": predictive_disagreement,
        "normalized_predictive_disagreement": normalized_predictive_disagreement,
        "ambiguity": ambiguity,
        "fgsm_results": fgsm_results,
        "bim_results": bim_results,
        "pgd_results": pgd_results,
    }


def calculate_ece(stats: dict) -> float:
    """
    Вычисляет Expected Calibration Error (ECE).
    
    ECE измеряет разницу между уверенностью модели и её точностью.
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
    return float(ece)


def calculate_nll(stats: dict) -> float:
    """
    Вычисляет Negative Log-Likelihood (NLL) ансамбля.
    
    Чем ниже NLL, тем лучше калибровка модели.
    """
    sum_nll = stats["sum_nll"]
    total = stats["total"]
    return sum_nll / total if total > 0 else float("nan")


def calculate_oracle_nll(stats: dict) -> float:
    """
    Вычисляет Oracle NLL - минимальный NLL среди всех моделей ансамбля.
    
    Показывает потенциал ансамбля при оптимальном выборе модели для каждого примера.
    """
    sum_oracle_nll = stats["sum_oracle_nll"]
    total = stats["total"]
    return sum_oracle_nll / total if total > 0 else float("nan")


def calculate_brier_score(stats: dict) -> float:
    """
    Вычисляет Brier Score.
    
    Brier Score - квадратичная оценка качества вероятностных предсказаний.
    Чем ниже, тем лучше.
    """
    sum_brier_score = stats["sum_brier_score"]
    total = stats["total"]
    return sum_brier_score / total if total > 0 else float("nan")


def denormalize(tensor, mean, std):
    """
    Денормализует тензор: x_original = x_norm * std + mean
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return tensor * std + mean


def normalize(tensor, mean, std):
    """
    Нормализует тензор: x_norm = (x - mean) / std
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return (tensor - mean) / std


def _ensemble_forward(models, images_norm, labels=None):
    """
    Вычисление средних логитов ансамбля.
    
    Args:
        models: Список моделей
        images_norm: Нормализованные изображения
        labels: Метки (опционально)
    
    Returns:
        avg_logits или (avg_logits, loss)
    """
    logits_sum = None
    for model in models:
        model.eval()
        logits = model(images_norm)
        if logits_sum is None:
            logits_sum = torch.zeros_like(logits)
        logits_sum += logits
    
    avg_logits = logits_sum / len(models)
    
    if labels is not None:
        loss = F.cross_entropy(avg_logits, labels)
        return avg_logits, loss
    return avg_logits


# ===== FGSM =====
def adversarial_attack_fgsm(
    models, test_loader, epsilon_list, device, mean, std, developer_mode=False
):
    """
    FGSM (Fast Gradient Sign Method) атака для ансамбля моделей.
    
    Однократная атака по знаку градиента функции потерь.
    Adversarial примеры генерируются для обмана ансамбля.
    """
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)

    results = {}
    for epsilon in epsilon_list:
        total = 0
        correct_ensemble = 0
        correct_models = [0 for _ in models]

        for images, labels in tqdm(
            test_loader, desc=f"FGSM (eps={epsilon})", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            images_denorm = torch.clamp(images * std_tensor + mean_tensor, 0, 1)

            if epsilon == 0:
                # Без атаки - просто оценка на чистых данных
                adv_images = images_denorm
            else:
                # Генерация adversarial примера
                images_adv = images_denorm.clone().detach().requires_grad_(True)
                
                # Очистка градиентов моделей
                for model in models:
                    if hasattr(model, 'zero_grad'):
                        model.zero_grad()
                
                avg_logits, loss = _ensemble_forward(
                    models, (images_adv - mean_tensor) / std_tensor, labels
                )
                loss.backward()

                with torch.no_grad():
                    grad_sign = images_adv.grad.sign()
                    adv_images = torch.clamp(images_denorm + epsilon * grad_sign, 0, 1)

            # Оценка на adversarial примерах
            perturbed_norm = (adv_images - mean_tensor) / std_tensor
            total += labels.size(0)

            with torch.no_grad():
                avg_output = None
                for idx, model in enumerate(models):
                    output = model(perturbed_norm).softmax(dim=1)
                    _, pred = output.max(1)
                    correct_models[idx] += (pred == labels).sum().item()
                    if avg_output is None:
                        avg_output = torch.zeros_like(output)
                    avg_output += output

                avg_output /= len(models)
                _, ens_pred = avg_output.max(1)
                correct_ensemble += (ens_pred == labels).sum().item()

            if developer_mode:
                break

        ensemble_acc = correct_ensemble / total * 100.0 if total > 0 else 0.0
        model_accs = [
            correct / total * 100.0 if total > 0 else 0.0 for correct in correct_models
        ]
        results[epsilon] = {
            "ensemble_acc": ensemble_acc,
            "model_accs": model_accs,
            "attack_type": "FGSM",
            "num_steps": 1,
        }

    return results


# ===== BIM (Iterative FGSM) =====
def adversarial_attack_bim(
    models,
    test_loader,
    epsilon_list,
    device,
    mean,
    std,
    num_steps=10,
    developer_mode=False,
):
    """
    BIM (Basic Iterative Method) атака для ансамбля.
    Оптимизированная версия с исправлением утечек памяти.
    """
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    results = {}
    for epsilon in epsilon_list:
        total = 0
        correct_ensemble = 0
        correct_models = [0 for _ in models]
        alpha = epsilon / num_steps if epsilon > 0 else 0

        for images, labels in tqdm(
            test_loader, desc=f"BIM (eps={epsilon:.4f})", leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            
            images_denorm = images * std_tensor + mean_tensor
            images_denorm = images_denorm.clamp(0, 1)
            
            if epsilon == 0:
                adv_images = images_denorm
            else:
                adv_images = images_denorm.clone().detach()
                
                for step in range(num_steps):
                    adv_images_step = adv_images.clone().requires_grad_(True)
                    
                    adv_images_norm = (adv_images_step - mean_tensor) / std_tensor
                    
                    logits_sum = None
                    for model in models:
                        logits = model(adv_images_norm)
                        if logits_sum is None:
                            logits_sum = logits
                        else:
                            logits_sum = logits_sum + logits
                    
                    avg_logits = logits_sum / len(models)
                    loss = F.cross_entropy(avg_logits, labels)
                    loss.backward()
                    
                    with torch.no_grad():
                        grad_sign = adv_images_step.grad.sign()
                        adv_images = adv_images + alpha * grad_sign
                        
                        perturbation = adv_images - images_denorm
                        perturbation = perturbation.clamp(-epsilon, epsilon)
                        adv_images = (images_denorm + perturbation).clamp(0, 1)

            # Оценка
            with torch.no_grad():
                perturbed_norm = (adv_images - mean_tensor) / std_tensor
                
                outputs = []
                for idx, model in enumerate(models):
                    output = model(perturbed_norm)
                    outputs.append(output)
                    pred = output.argmax(dim=1)
                    correct_models[idx] += (pred == labels).sum().item()
                
                avg_output = torch.stack(outputs).mean(dim=0).softmax(dim=1)
                ens_pred = avg_output.argmax(dim=1)
                correct_ensemble += (ens_pred == labels).sum().item()
            
            total += batch_size

            if developer_mode:
                break

        ensemble_acc = correct_ensemble / total * 100.0 if total > 0 else 0.0
        model_accs = [
            correct / total * 100.0 if total > 0 else 0.0 for correct in correct_models
        ]
        results[epsilon] = {
            "ensemble_acc": ensemble_acc,
            "model_accs": model_accs,
            "attack_type": "BIM",
            "num_steps": num_steps,
        }

    return results


def adversarial_attack_pgd(
    models,
    test_loader,
    epsilon_list,
    device,
    mean,
    std,
    num_steps=10,
    developer_mode=False,
):
    """
    PGD (Projected Gradient Descent) атака для ансамбля.
    Оптимизированная версия с исправлением утечек памяти.
    """
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    results = {}
    for epsilon in epsilon_list:
        total = 0
        correct_ensemble = 0
        correct_models = [0 for _ in models]
        alpha = epsilon / num_steps if epsilon > 0 else 0

        for images, labels in tqdm(
            test_loader, desc=f"PGD (eps={epsilon:.4f})", leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            
            images_denorm = images * std_tensor + mean_tensor
            images_denorm = images_denorm.clamp(0, 1)

            if epsilon == 0:
                adv_images = images_denorm
            else:
                # Случайная инициализация
                noise = torch.empty_like(images_denorm).uniform_(-epsilon, epsilon)
                adv_images = (images_denorm + noise).clamp(0, 1).detach()

                for step in range(num_steps):
                    adv_images_step = adv_images.clone().requires_grad_(True)
                    
                    adv_images_norm = (adv_images_step - mean_tensor) / std_tensor
                    
                    logits_sum = None
                    for model in models:
                        logits = model(adv_images_norm)
                        if logits_sum is None:
                            logits_sum = logits
                        else:
                            logits_sum = logits_sum + logits
                    
                    avg_logits = logits_sum / len(models)
                    loss = F.cross_entropy(avg_logits, labels)
                    loss.backward()
                    
                    with torch.no_grad():
                        grad_sign = adv_images_step.grad.sign()
                        adv_images = adv_images + alpha * grad_sign
                        
                        perturbation = adv_images - images_denorm
                        perturbation = perturbation.clamp(-epsilon, epsilon)
                        adv_images = (images_denorm + perturbation).clamp(0, 1)

            # Оценка
            with torch.no_grad():
                perturbed_norm = (adv_images - mean_tensor) / std_tensor
                
                outputs = []
                for idx, model in enumerate(models):
                    output = model(perturbed_norm)
                    outputs.append(output)
                    pred = output.argmax(dim=1)
                    correct_models[idx] += (pred == labels).sum().item()
                
                avg_output = torch.stack(outputs).mean(dim=0).softmax(dim=1)
                ens_pred = avg_output.argmax(dim=1)
                correct_ensemble += (ens_pred == labels).sum().item()
            
            total += batch_size

            if developer_mode:
                break

        ensemble_acc = correct_ensemble / total * 100.0 if total > 0 else 0.0
        model_accs = [
            correct / total * 100.0 if total > 0 else 0.0 for correct in correct_models
        ]
        results[epsilon] = {
            "ensemble_acc": ensemble_acc,
            "model_accs": model_accs,
            "attack_type": "PGD",
            "num_steps": num_steps,
        }

    return results


# ===== Обертка =====
def adversarial_attack(
    models,
    test_loader,
    epsilon_list=[0.01, 0.02, 0.05],
    device=None,
    mean=None,
    std=None,
    attack_type="FGSM",
    num_steps=10,
    developer_mode=False,
):
    """
    Унифицированная обертка для выбора типа adversarial атаки.
    
    Args:
        models: Список моделей ансамбля
        test_loader: DataLoader для тестовых данных
        epsilon_list: Список значений epsilon для атаки (размер возмущения)
        device: Устройство (CPU/GPU)
        mean: Средние значения для нормализации (список или число)
        std: Стандартные отклонения для нормализации (список или число)
        attack_type: Тип атаки ("FGSM", "BIM", "PGD")
        num_steps: Количество итераций для BIM/PGD
        developer_mode: Режим разработки (обработка только одного батча)
    
    Returns:
        Словарь с результатами атаки для каждого epsilon:
        {
            epsilon: {
                "ensemble_acc": точность ансамбля на adversarial примерах,
                "model_accs": точности отдельных моделей (transferability),
                "attack_type": тип атаки,
                "num_steps": количество итераций
            }
        }
    
    Note:
        - Adversarial примеры генерируются для обмана АНСАМБЛЯ (white-box атака на ансамбль)
        - model_accs показывают transferability: насколько хорошо adversarial примеры,
          созданные для ансамбля, переносятся на отдельные модели
        - epsilon=0 соответствует оценке на чистых данных (baseline)
    """
    attack_type = attack_type.upper()

    if attack_type == "FGSM":
        return adversarial_attack_fgsm(
            models, test_loader, epsilon_list, device, mean, std, developer_mode
        )
    elif attack_type == "BIM":
        return adversarial_attack_bim(
            models,
            test_loader,
            epsilon_list,
            device,
            mean,
            std,
            num_steps,
            developer_mode,
        )
    elif attack_type == "PGD":
        return adversarial_attack_pgd(
            models,
            test_loader,
            epsilon_list,
            device,
            mean,
            std,
            num_steps,
            developer_mode,
        )
    else:
        raise ValueError(
            f"Неизвестный тип атаки: {attack_type}. Ожидается FGSM, BIM или PGD."
        )