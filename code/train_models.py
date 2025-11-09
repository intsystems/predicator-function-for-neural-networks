import os
import json
import shutil
import re
import argparse
from pathlib import Path
import random
from typing import List, Tuple, Optional, Dict, Any
import time

import numpy as np
import torch
import nni
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from nni.nas.evaluator.pytorch import DataLoader, Lightning, Trainer
from nni.nas.space import model_context
from tqdm import tqdm

import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

# Импорты из проекта
from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.darts_classification_module import DartsClassificationModule
from dependencies.train_config import TrainConfig
from dependencies.data_generator import generate_arch_dicts
from dependencies.metrics import (
    collect_ensemble_stats,
    calculate_nll,
    calculate_ece,
    calculate_oracle_nll,
    calculate_brier_score,
)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===


def load_json_from_directory(directory_path: Path) -> List[dict]:
    """
    Загружает все JSON-файлы из директории и её поддиректорий.
    Добавляет поле 'id' из имени файла, если отсутствует.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    json_data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "id" not in data:
                        match = re.search(r"(\d+)\.json$", file)
                        if match:
                            data["id"] = int(match.group(1))
                    json_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
    return json_data


class Cutout:
    """Apply cutout to an image tensor."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be Tensor. Got {type(img)}")

        device = img.device
        c, h, w = img.size()
        mask = torch.ones((h, w), dtype=torch.float32, device=device)

        y = torch.randint(0, h, (1,), device=device).item()
        x = torch.randint(0, w, (1,), device=device).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask[y1:y2, x1:x2] = 0.0
        mask = mask.unsqueeze(0).expand_as(img)
        return img * mask


def duplicate_channel(x: torch.Tensor) -> torch.Tensor:
    """Дублирует одноканальное изображение в 3 канала."""
    return x.repeat(3, 1, 1)


# === ИНФОРМАЦИЯ О ДАТАСЕТАХ ===


class DatasetsInfo:
    """Конфигурация датасетов: трансформации, нормализация, классы."""

    DATASETS = {
        "cifar10": {
            "key": "cifar",
            "class": CIFAR10,
            "num_classes": 10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                    Cutout(16),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            ),
        },
        "cifar100": {
            "key": "cifar100",
            "class": CIFAR100,
            "num_classes": 100,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2673, 0.2564, 0.2762],
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]
                    ),
                    Cutout(16),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]
                    ),
                ]
            ),
        },
        "fashionmnist": {
            "key": "cifar",
            "class": FashionMNIST,
            "num_classes": 10,
            "mean": [0.2860406] * 3,
            "std": [0.35302424] * 3,
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Lambda(duplicate_channel),
                    transforms.Normalize(mean=[0.2860406] * 3, std=[0.35302424] * 3),
                    Cutout(12),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(duplicate_channel),
                    transforms.Normalize(mean=[0.2860406] * 3, std=[0.35302424] * 3),
                ]
            ),
        },
    }

    @classmethod
    def get(cls, dataset_name: str) -> Dict[str, Any]:
        """Возвращает конфигурацию датасета."""
        dataset_name = dataset_name.lower()
        if dataset_name not in cls.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}"
            )
        return cls.DATASETS[dataset_name]


# === ОСНОВНОЙ КЛАСС ===


class DiversityNESRunner:
    def __init__(self, config: TrainConfig, info: Dict[str, Any]):
        self.config = config
        self.models: List[Optional[torch.nn.Module]] = []
        self.dataset_key = info["key"]
        self.dataset_cls = info["class"]
        self.num_classes = info["num_classes"]
        self.MEAN = info["mean"]
        self.STD = info["std"]
        self.img_size = info["img_size"]
        self.train_transform = info["train_transform"]
        self.test_transform = info["test_transform"]

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

    def get_data_loaders(
        self, batch_size: Optional[int] = None, seed: Optional[int] = None
    ):
        """Создаёт загрузчики данных."""
        bs = batch_size or self.config.batch_size_final
        dataset_cls = self.dataset_cls

        train_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=True,
            download=True,
            transform=self.train_transform,
        )
        test_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=False,
            download=True,
            transform=self.test_transform,
        )

        num_samples = len(train_data)
        indices = list(range(num_samples))
        np.random.seed(seed or self.config.seed)
        np.random.shuffle(indices)

        split = int(num_samples * self.config.train_size_final)
        train_subset = Subset(train_data, indices[:split])
        train_loader = DataLoader(
            train_subset, batch_size=bs, num_workers=00, shuffle=True
        )

        split_valid = int(num_samples * self.config.train_size_final)
        valid_subset = Subset(train_data, indices[split_valid:])
        valid_loader = DataLoader(
            valid_subset, batch_size=bs, num_workers=0, shuffle=False
        )

        test_loader = DataLoader(
            test_data, batch_size=bs, num_workers=20, shuffle=False
        )
        return train_loader, valid_loader, test_loader

    @staticmethod
    def _custom_weight_init(module: torch.nn.Module) -> None:
        """Инициализация весов для линейных и сверточных слоёв."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def train_model(
        self,
        architecture: dict,
        train_loader,
        valid_loader,
        model_id: int,
    ) -> Optional[torch.nn.Module]:
        """Обучает одну модель по архитектуре."""
        seed = self.config.seed + model_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            with model_context(architecture):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=self.dataset_key,
                )
            model = model.to(self.device)
            model.apply(self._custom_weight_init)

            evaluator = Lightning(
                DartsClassificationModule(
                    learning_rate=self.config.lr_start_final,
                    weight_decay=self.config.weight_decay,
                    auxiliary_loss_weight=self.config.auxiliary_loss_weight,
                    max_epochs=self.config.n_epochs_final,
                    num_classes=self.num_classes,
                    lr_final=self.config.lr_end_final,
                    label_smoothing=0.15,
                    optimizer=self.config.optimizer,
                ),
                trainer=Trainer(
                    gradient_clip_val=5.0,
                    max_epochs=self.config.n_epochs_final,
                    fast_dev_run=self.config.developer_mode,
                    accelerator="gpu" if self.device.type == "cuda" else "cpu",
                    devices=1,
                    strategy="auto",
                    enable_progress_bar=True,
                    precision="bf16-mixed",
                    benchmark=True,
                ),
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )
            evaluator.fit(model)
            model = model.to(self.device)

            return model

        except Exception as e:
            print(f"Error training model {model_id}: {e}")
            return None

    def evaluate_and_save_results(
        self,
        model: torch.nn.Module,
        architecture: dict,
        data_loader,
        folder_name: str,
        model_id: int,
        mode: str = "classes",
    ) -> None:
        """Оценивает модель и сохраняет предсказания."""
        device = next(model.parameters()).device
        model.eval()
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).softmax(dim=1)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                preds = (
                    outputs.cpu().tolist()
                    if mode == "logits"
                    else predicted.cpu().tolist()
                )
                predictions.extend(preds)

        accuracy = correct / total
        result = {
            "architecture": architecture,
            "valid_predictions": predictions,
            "valid_accuracy": accuracy,
        }

        folder = Path(folder_name)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"model_{model_id}.json"

        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results for model_{model_id} saved to {file_path}")

    def finalize_ensemble_evaluation(
        self, stats: Optional[Dict[str, Any]], file_name: str = "ensemble_results"
    ) -> Tuple[Optional[float], Optional[List[float]], Optional[float]]:
        """
        Финализирует оценку ансамбля: вычисляет метрики и сохраняет результаты.

        Args:
            stats: Словарь со статистикой из collect_ensemble_stats
            file_name: Базовое имя файла для сохранения результатов

        Returns:
            Tuple: (ensemble_accuracy, model_accuracies, ece)
        """
        if stats is None:
            print("No stats provided for evaluation.")
            return None, None, None

        # Извлечение базовых метрик
        total = stats["total"]
        correct_ensemble = stats["correct_ensemble"]
        correct_models = stats["correct_models"]
        num_models = stats["num_models"]

        ensemble_acc = 100.0 * correct_ensemble / total if total > 0 else 0.0
        model_accs = [100.0 * c / total if total > 0 else 0.0 for c in correct_models]
        avg_model_acc = stats["avg_model_accuracy"] * 100.0

        # Вычисление метрик
        ece = calculate_ece(stats)
        nll = calculate_nll(stats)
        oracle_nll = calculate_oracle_nll(stats)
        brier_score = calculate_brier_score(stats)

        # Метрики разнообразия
        predictive_disagreement = stats.get("predictive_disagreement", float("nan"))
        normalized_predictive_disagreement = stats.get(
            "normalized_predictive_disagreement", float("nan")
        )
        ambiguity = stats.get("ambiguity", float("nan"))

        # Adversarial атаки
        fgsm_results = stats.get("fgsm_results", {})
        bim_results = stats.get("bim_results", {})
        pgd_results = stats.get("pgd_results", {})

        # -------- Форматированный вывод в консоль ----------
        print("=" * 60)
        print("ENSEMBLE EVALUATION RESULTS")
        print("=" * 60)

        print("\n--- Basic Metrics ---")
        print(f"Number of models:                    {num_models}")
        print(f"Total samples:                       {total}")
        print(f"Ensemble Top-1 Accuracy:             {ensemble_acc:.2f}%")
        print(f"Average Model Top-1 Accuracy:        {avg_model_acc:.2f}%")

        print("\n--- Individual Model Accuracies ---")
        for i, acc in enumerate(model_accs):
            print(f"  Model {i+1} Top-1 Accuracy:        {acc:.2f}%")

        print("\n--- Calibration Metrics ---")
        print(f"Expected Calibration Error (ECE):    {ece:.4f}")
        print(f"Negative Log-Likelihood (NLL):       {nll:.4f}")
        print(f"Oracle NLL:                          {oracle_nll:.4f}")
        print(f"Brier Score:                         {brier_score:.4f}")

        print("\n--- Diversity Metrics ---")
        print(f"Ambiguity (Ensemble Benefit):        {ambiguity:.4f}")
        print(f"Predictive Disagreement:             {predictive_disagreement:.4f}")
        print(
            f"Normalized Pred. Disagreement:       {normalized_predictive_disagreement:.4f}"
        )

        # === FGSM ===
        print("\n" + "=" * 60)
        print("ADVERSARIAL ATTACK RESULTS (FGSM)")
        print("=" * 60)
        if fgsm_results:
            for eps, eps_info in sorted(fgsm_results.items()):
                print(f"\nEpsilon = {eps:.4f}")
                print(
                    f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%"
                )
                for i, acc in enumerate(eps_info["model_accs"]):
                    print(f"  Model {i+1} accuracy:              {acc:.2f}%")
        else:
            print("FGSM results not available.")

        # === BIM ===
        print("\n" + "=" * 60)
        print("ADVERSARIAL ATTACK RESULTS (BIM)")
        print("=" * 60)
        if bim_results:
            num_steps_bim = (
                list(bim_results.values())[0].get("num_steps", "N/A")
                if bim_results
                else "N/A"
            )
            print(f"Number of iterations: {num_steps_bim}")
            for eps, eps_info in sorted(bim_results.items()):
                print(f"\nEpsilon = {eps:.4f}")
                print(
                    f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%"
                )
                for i, acc in enumerate(eps_info["model_accs"]):
                    print(f"  Model {i+1} accuracy:              {acc:.2f}%")
        else:
            print("BIM results not available.")

        # === PGD ===
        print("\n" + "=" * 60)
        print("ADVERSARIAL ATTACK RESULTS (PGD)")
        print("=" * 60)
        if pgd_results:
            num_steps_pgd = (
                list(pgd_results.values())[0].get("num_steps", "N/A")
                if pgd_results
                else "N/A"
            )
            print(f"Number of iterations: {num_steps_pgd}")
            for eps, eps_info in sorted(pgd_results.items()):
                print(f"\nEpsilon = {eps:.4f}")
                print(
                    f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%"
                )
                for i, acc in enumerate(eps_info["model_accs"]):
                    print(f"  Model {i+1} accuracy:              {acc:.2f}%")
        else:
            print("PGD results not available.")

        print("\n" + "=" * 60)

        # --------- Запись в файл --------------
        output_path = Path(self.config.output_path)
        output_path.mkdir(exist_ok=True)
        experiment_num = self._get_free_file_index(str(output_path), file_name)
        out_file = output_path / f"{file_name}_{experiment_num}.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("ENSEMBLE EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")

            f.write("\n--- Basic Metrics ---\n")
            f.write(f"Number of models:                    {num_models}\n")
            f.write(f"Total samples:                       {total}\n")
            f.write(f"Ensemble Top-1 Accuracy:             {ensemble_acc:.2f}%\n")
            f.write(f"Average Model Top-1 Accuracy:        {avg_model_acc:.2f}%\n")

            f.write("\n--- Individual Model Accuracies ---\n")
            for i, acc in enumerate(model_accs):
                f.write(f"  Model {i+1} Top-1 Accuracy:        {acc:.2f}%\n")

            f.write("\n--- Calibration Metrics ---\n")
            f.write(f"Expected Calibration Error (ECE):    {ece:.4f}\n")
            f.write(f"Negative Log-Likelihood (NLL):       {nll:.4f}\n")
            f.write(f"Oracle NLL:                          {oracle_nll:.4f}\n")
            f.write(f"Brier Score:                         {brier_score:.4f}\n")

            f.write("\n--- Diversity Metrics ---\n")
            f.write(f"Ambiguity (Ensemble Benefit):        {ambiguity:.4f}\n")
            f.write(
                f"Predictive Disagreement:             {predictive_disagreement:.4f}\n"
            )
            f.write(
                f"Normalized Pred. Disagreement:       {normalized_predictive_disagreement:.4f}\n"
            )

            f.write("\n" + "=" * 60 + "\n")
            f.write("ADVERSARIAL ATTACK RESULTS (FGSM)\n")
            f.write("=" * 60 + "\n")
            if fgsm_results:
                for eps, eps_info in sorted(fgsm_results.items()):
                    f.write(f"\nEpsilon = {eps:.4f}\n")
                    f.write(
                        f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%\n"
                    )
                    for i, acc in enumerate(eps_info["model_accs"]):
                        f.write(f"  Model {i+1} accuracy:              {acc:.2f}%\n")
            else:
                f.write("FGSM results not available.\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("ADVERSARIAL ATTACK RESULTS (BIM)\n")
            f.write("=" * 60 + "\n")
            if bim_results:
                num_steps_bim = (
                    list(bim_results.values())[0].get("num_steps", "N/A")
                    if bim_results
                    else "N/A"
                )
                f.write(f"Number of iterations: {num_steps_bim}\n")
                for eps, eps_info in sorted(bim_results.items()):
                    f.write(f"\nEpsilon = {eps:.4f}\n")
                    f.write(
                        f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%\n"
                    )
                    for i, acc in enumerate(eps_info["model_accs"]):
                        f.write(f"  Model {i+1} accuracy:              {acc:.2f}%\n")
            else:
                f.write("BIM results not available.\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("ADVERSARIAL ATTACK RESULTS (PGD)\n")
            f.write("=" * 60 + "\n")
            if pgd_results:
                num_steps_pgd = (
                    list(pgd_results.values())[0].get("num_steps", "N/A")
                    if pgd_results
                    else "N/A"
                )
                f.write(f"Number of iterations: {num_steps_pgd}\n")
                for eps, eps_info in sorted(pgd_results.items()):
                    f.write(f"\nEpsilon = {eps:.4f}\n")
                    f.write(
                        f"  Ensemble accuracy:                 {eps_info['ensemble_acc']:.2f}%\n"
                    )
                    for i, acc in enumerate(eps_info["model_accs"]):
                        f.write(f"  Model {i+1} accuracy:              {acc:.2f}%\n")
            else:
                f.write("PGD results not available.\n")

            # Дополнительная информация для анализа
            f.write("\n" + "=" * 60 + "\n")
            f.write("ADDITIONAL INFORMATION\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"Ensemble improvement over avg model: {ensemble_acc - avg_model_acc:.2f}%\n"
            )
            f.write(f"NLL improvement (Oracle - Ensemble): {oracle_nll - nll:.4f}\n")

            # Робастность к атакам
            if fgsm_results:
                max_eps_fgsm = max(fgsm_results.keys())
                fgsm_degradation = (
                    ensemble_acc - fgsm_results[max_eps_fgsm]["ensemble_acc"]
                )
                f.write(
                    f"FGSM accuracy drop (eps={max_eps_fgsm:.4f}):    {fgsm_degradation:.2f}%\n"
                )

            if bim_results:
                max_eps_bim = max(bim_results.keys())
                bim_degradation = (
                    ensemble_acc - bim_results[max_eps_bim]["ensemble_acc"]
                )
                f.write(
                    f"BIM accuracy drop (eps={max_eps_bim:.4f}):     {bim_degradation:.2f}%\n"
                )

            if pgd_results:
                max_eps_pgd = max(pgd_results.keys())
                pgd_degradation = (
                    ensemble_acc - pgd_results[max_eps_pgd]["ensemble_acc"]
                )
                f.write(
                    f"PGD accuracy drop (eps={max_eps_pgd:.4f}):     {pgd_degradation:.2f}%\n"
                )

        print(f"\nResults saved to: {out_file}")
        return ensemble_acc, model_accs, ece

    def _get_free_file_index(self, path: str, file_name: str) -> int:
        """Находит свободный индекс для имени файла."""
        experiment_num = 0
        while (Path(path) / f"{file_name}_{experiment_num}.txt").exists():
            experiment_num += 1
        return experiment_num

    def _get_free_file_index_dir(self, path: str, dir_name: str) -> int:
        """Находит свободный индекс для имени папки."""
        experiment_num = 1
        while (Path(path) / f"{dir_name}_{experiment_num}").exists():
            experiment_num += 1
        return experiment_num

    def get_latest_index_from_dir(self, base_path: Optional[Path] = None) -> int:
        """Находит последний индекс в директориях models_json_*."""
        if base_path is None:
            base_path = Path(self.config.best_models_save_path)

        candidates = [p for p in base_path.glob("models_json_*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No models_json_* directories in {base_path}")

        indices = [
            int(m.group(1))
            for p in candidates
            if (m := re.search(r"models_json_(\d+)", str(p)))
        ]
        if not indices:
            raise ValueError("No valid indices in models_json_* directories")

        return max(indices)

    @staticmethod
    def train_single_model_process(
        architecture: dict,
        model_id: int,
        physical_gpu_id: int,
        config: TrainConfig,
        dataset_key: str,
        num_classes: int,
        archs_path,
        pth_path,
    ):
        """Целевой процесс для параллельного обучения."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        max_retries = 99999
        retry_delay = 300
        for attempt in range(max_retries + 1):
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    torch.cuda.empty_cache()

                torch.set_float32_matmul_precision("high")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                runner = DiversityNESRunner(config, DatasetsInfo.get(config.dataset_name.lower()))
                train_loader, valid_loader, test_loader = runner.get_data_loaders()

                model = runner.train_model(architecture, train_loader, None, model_id)
                model = model.to(device)

                torch.save(model.state_dict(), pth_path / f"model_{model_id}.pth")
                print(f"[GPU {physical_gpu_id}] ✅ Model {model_id} saved to {pth_path}")

                eval_loader = test_loader if config.evaluate_ensemble_flag else valid_loader
                runner.evaluate_and_save_results(
                    model, architecture, eval_loader, str(archs_path), model_id=model_id
                )
                break

            except Exception as e:
                print(f"[GPU {physical_gpu_id}] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} sec...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Model {model_id} failed after {max_retries} attempts.")
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


    def fill_models_list(self, archs_path, pth_path) -> None:
        """Загружает обученные модели из .pth и .json."""
        if not archs_path.exists() or not pth_path.exists():
            raise FileNotFoundError(f"Missing directories: {archs_path}, {pth_path}")

        self.models = []
        json_files = sorted(archs_path.glob("model_*.json"))
        print(f"Found {len(json_files)} models to load.")

        for json_file in json_files:
            match = re.search(r"model_(\d+)\.json", json_file.name)
            if not match:
                print(f"Skipping invalid filename: {json_file.name}")
                continue
            model_id = int(match.group(1))

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                arch = data["architecture"]

                with model_context(arch):
                    model = DartsSpace(
                        width=self.config.width,
                        num_cells=self.config.num_cells,
                        dataset=self.dataset_key,
                    )

                pth_file = pth_path / f"model_{model_id}.pth"
                if not pth_file.exists():
                    print(f"Weights not found: {pth_file}")
                    continue

                state_dict = torch.load(pth_file, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                self.models.append(model)
                print(f"✅ Model {model_id} loaded")

            except Exception as e:
                print(f"❌ Failed to load {json_file}: {e}")

        print(f"✅ Loaded {len(self.models)} models.")

    def run_pretrained(self, index: int) -> None:
        """Загружает и параллельно оценивает предобученные модели по индексу."""
        root_dir = Path(self.config.best_models_save_path)
        json_dir = root_dir / f"trained_models_archs_{index}"
        pth_dir = root_dir / f"trained_models_pth_{index}"

        if not json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")
        if not pth_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pth_dir}")

        print(f"Using JSON from: {json_dir}, PTH from: {pth_dir}")
        arch_dicts = load_json_from_directory(json_dir)
        if not arch_dicts:
            raise RuntimeError("No architectures found")

        # Подготовка директорий для сохранения результатов
        eval_output_dir = (
            Path(self.config.output_path) / f"trained_models_archs_{index}"
        )
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Получаем список моделей: (arch, model_id, pth_path)
        model_tasks = []
        for entry in arch_dicts:
            arch = entry["architecture"]
            model_id = entry.get("id")
            if model_id is None:
                raise ValueError("Missing 'id' in architecture")
            pth_file = pth_dir / f"model_{model_id}.pth"
            if not pth_file.exists():
                print(f"⚠️ Skipping model {model_id}: weights not found at {pth_file}")
                continue
            model_tasks.append((arch, model_id, pth_file))

        if not model_tasks:
            print("⚠️ No valid models to evaluate.")
            return

        n_models = len(model_tasks)
        available_gpus = self._get_available_gpus()
        n_gpus = len(available_gpus)
        max_per_gpu = self.config.max_per_gpu

        print(f"Available GPUs: {available_gpus}")
        print(f"Evaluating {n_models} models, up to {max_per_gpu} per GPU")

        # Запуск параллельных процессов
        processes = []
        active_processes = []  # будем хранить (process, model_id) для отслеживания

        for idx, (arch, model_id, pth_path) in enumerate(model_tasks):
            gpu_idx = (idx // max_per_gpu) % n_gpus
            physical_gpu_id = available_gpus[gpu_idx]

            p = mp.get_context("spawn").Process(
                target=self._evaluate_single_model_process,
                args=(
                    arch,
                    model_id,
                    pth_path,
                    str(eval_output_dir),
                    physical_gpu_id,
                    self.config,
                    self.dataset_key,
                    self.num_classes,
                ),
            )
            p.start()
            active_processes.append((p, model_id))
            print(f"Started evaluation of model {model_id} on GPU {physical_gpu_id}")

            # Ограничение активных процессов
            while len(active_processes) >= n_gpus * max_per_gpu:
                # Проверяем завершённые процессы
                for proc, mid in active_processes[:]:
                    if not proc.is_alive():
                        proc.join()
                        active_processes.remove((proc, mid))
                        break
                time.sleep(0.5)

        # Дожидаемся всех оставшихся + прогресс-бар
        completed = 0
        with tqdm(total=n_models, desc="Evaluating models") as pbar:
            while active_processes:
                # Проверяем все процессы на завершение
                for proc, mid in active_processes[:]:
                    if not proc.is_alive():
                        proc.join()
                        active_processes.remove((proc, mid))
                        completed += 1
                        pbar.update(1)
                time.sleep(0.5)  # небольшая пауза, чтобы не грузить CPU

        print(f"✅ All models from index {index} evaluated!")

        # Если нужно — оценить ансамбль
        if self.config.evaluate_ensemble_flag:
            print("📊 Evaluating ensemble for pretrained models...")
            self.models = []
            self._load_models_from_index(index)
            if self.models:
                _, _, test_loader = self.get_data_loaders()
                stats = collect_ensemble_stats(
                    models=self.models,
                    device=self.device,
                    test_loader=test_loader,
                    n_ece_bins=self.config.n_ece_bins,
                    developer_mode=self.config.developer_mode,
                    mean=self.MEAN,
                    std=self.STD,
                )
                self.finalize_ensemble_evaluation(stats, f"ensemble_results_{index}")
            else:
                print("❌ No models loaded for ensemble evaluation.")

    @staticmethod
    def _evaluate_single_model_process(
        architecture: dict,
        model_id: int,
        pth_path: Path,
        eval_output_dir: str,
        physical_gpu_id: int,
        config: TrainConfig,
        dataset_key: str,
        num_classes: int,
    ):
        """Процесс оценки одной предобученной модели."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("high")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Воссоздаём runner для получения трансформаций и загрузчиков
            info = DatasetsInfo.get(config.dataset_name.lower())
            runner = DiversityNESRunner(config, info)
            _, valid_loader, test_loader = runner.get_data_loaders()

            if config.evaluate_ensemble_flag:
                eval_loader = test_loader
            else:
                eval_loader = valid_loader

            # Загружаем модель
            with model_context(architecture):
                model = DartsSpace(
                    width=config.width,
                    num_cells=config.num_cells,
                    dataset=dataset_key,
                )
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)

            # Оцениваем
            runner.evaluate_and_save_results(
                model,
                architecture,
                eval_loader,
                folder_name=eval_output_dir,
                model_id=model_id,
                mode="class",
            )

        except Exception as e:
            print(f"❌ Error evaluating model {model_id} on GPU {physical_gpu_id}: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_models_from_index(self, index: int) -> None:
        """Загружает все модели из указанного индекса для ансамбля."""
        root_dir = Path(self.config.best_models_save_path)
        json_dir = root_dir / f"trained_models_archs_{index}"
        pth_dir = root_dir / f"trained_models_pth_{index}"

        arch_dicts = load_json_from_directory(json_dir)
        self.models = []

        for entry in arch_dicts:
            arch = entry["architecture"]
            model_id = entry.get("id")
            if model_id is None:
                continue
            pth_file = pth_dir / f"model_{model_id}.pth"
            if not pth_file.exists():
                continue

            with model_context(arch):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=self.dataset_key,
                )
            state_dict = torch.load(pth_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            self.models.append(model)

    def run_all_pretrained(self) -> None:
        """Оценивает все сохранённые ансамбли."""
        root_dir = Path(self.config.best_models_save_path)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # last_index = self.get_latest_index_from_dir(root_dir)
        last_index = self._get_free_file_index_dir(root_dir, "trained_models_archs")
        print(f"Last index: {last_index}")
        for idx in range(1, last_index + 1):
            if (root_dir / f"trained_models_archs_{idx}").exists() and (
                root_dir / f"trained_models_pth_{idx}"
            ).exists():
                self.run_pretrained(idx)
                self.models = []  # очистка после каждого индекса
            else:
                print(f"Skipping index {idx}: missing directories.")

    def run(self) -> None:
        """Основной пайплайн: обучение или оценка."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("Loading architectures...")
        index_2_json_dir = {}
        latest_index = self.get_latest_index_from_dir()
        assert latest_index > 0, "No architectures found!"

        for cur_index in range(1, latest_index + 1):
            models_json_dir = (
                Path(self.config.best_models_save_path) / f"models_json_{cur_index}"
            )
            index_2_json_dir[cur_index] = models_json_dir

        archs_list = []  # its array of tuple (json_dir_index, arch_dict)
        index_2_n_archs = {}
        for cur_index in index_2_json_dir.keys():
            print(f"Loading architectures from {index_2_json_dir[cur_index]}")
            arch_dicts = load_json_from_directory(index_2_json_dir[cur_index])
            archs = [d["architecture"] for d in arch_dicts]
            archs_list.extend([(cur_index, arch) for arch in archs])
            index_2_n_archs[cur_index] = len(archs)

        assert archs_list, "No architectures to train!"

        n_models = len(archs_list)
        available_gpus = self._get_available_gpus()

        if not available_gpus:
            print("No GPUs available!")
            return

        n_gpus = len(available_gpus)
        max_per_gpu = self.config.max_per_gpu
        total_processes = n_models

        print(f"Available GPUs: {available_gpus}")
        print(f"Training {n_models} models, up to {max_per_gpu} per GPU")

        archs_and_pth_path_list = []
        output_path = Path(self.config.output_path)

        for cur_index in index_2_json_dir.keys():
            archs_path = output_path / f"trained_models_archs_{cur_index}"
            pth_path = output_path / f"trained_models_pth_{cur_index}"
            archs_and_pth_path_list.append((archs_path, pth_path))
            archs_path.mkdir(parents=True, exist_ok=True)
            pth_path.mkdir(parents=True, exist_ok=True)

        processes = []

        for idx, (cur_index, arch) in enumerate(archs_list):
            gpu_idx = (idx // max_per_gpu) % n_gpus
            physical_gpu_id = available_gpus[gpu_idx]
            model_id = idx % index_2_n_archs[cur_index]

            archs_path = output_path / f"trained_models_archs_{cur_index}"
            pth_path = output_path / f"trained_models_pth_{cur_index}"

            p = mp.get_context("spawn").Process(
                target=self.train_single_model_process,
                args=(
                    arch,
                    model_id,
                    physical_gpu_id,
                    self.config,
                    self.dataset_key,
                    self.num_classes,
                    archs_path,
                    pth_path,
                ),
            )
            p.start()
            processes.append(p)
            print(f"Started model {idx} on GPU {physical_gpu_id}")

            # Ограничение количества процессов
            while len(processes) >= n_gpus * max_per_gpu:
                time.sleep(2)
                for p in processes[:]:
                    if not p.is_alive():
                        p.join()
                        processes.remove(p)
                        break

        # Ожидание завершения всех процессов
        for p in processes:
            p.join()
        print("✅ All models trained!")

        # Оценка ансамбля
        if self.config.evaluate_ensemble_flag:
            print("📊 Evaluating ensemble...")
            for cur_index, (archs_path, pth_path) in enumerate(archs_and_pth_path_list):
                self.models = []
                self.fill_models_list(archs_path, pth_path)
                if self.models:
                    _, _, test_loader = self.get_data_loaders()
                    stats = collect_ensemble_stats(
                        models=self.models,
                        device=self.device,
                        test_loader=test_loader,
                        n_ece_bins=self.config.n_ece_bins,
                        developer_mode=self.config.developer_mode,
                        mean=self.MEAN,
                        std=self.STD,
                    )
                    self.finalize_ensemble_evaluation(
                        stats, f"ensemble_results_{cur_index}"
                    )
                else:
                    print("❌ No models to evaluate.")
        else:
            print("⏭️ Ensemble evaluation skipped.")

        print("🎉 Training and evaluation completed!")

    def _get_available_gpus(self) -> List[int]:
        """Определяет доступные GPU."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return [int(x.strip()) for x in cuda_visible.split(",")]
        return list(range(torch.cuda.device_count()))


# === ТОЧКА ВХОДА ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTS NAS Runner")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        required=True,
        help="Path to JSON file with TrainConfig parameters",
    )
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    params["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert params["seed"] != -1, "Seed must be set!"

    config = TrainConfig(**params)
    info = DatasetsInfo.get(config.dataset_name.lower())
    runner = DiversityNESRunner(config, info)

    if config.use_pretrained_models_for_ensemble:
        runner.run_all_pretrained()
    else:
        runner.run()
