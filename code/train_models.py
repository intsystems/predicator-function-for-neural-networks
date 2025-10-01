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
                    Cutout(16),
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
    ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
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

        split = int(num_samples * self.config.train_size)
        train_subset = Subset(train_data, indices[:split])
        train_loader = DataLoader(
            train_subset, batch_size=bs, num_workers=0, shuffle=True
        )

        split_valid = int(num_samples * max(0.9, self.config.train_size))
        valid_subset = Subset(train_data, indices[split_valid:])
        valid_loader = DataLoader(
            valid_subset, batch_size=bs, num_workers=0, shuffle=False
        )

        test_loader = DataLoader(test_data, batch_size=bs, num_workers=0, shuffle=False)
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
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
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

            tmp_valid_loader = (
                valid_loader if self.config.evaluate_ensemble_flag else None
            )

            evaluator = Lightning(
                DartsClassificationModule(
                    learning_rate=self.config.lr_start_final,
                    weight_decay=self.config.weight_decay,
                    auxiliary_loss_weight=self.config.auxiliary_loss_weight,
                    max_epochs=self.config.n_epochs_final,
                    num_classes=self.num_classes,
                    lr_final=self.config.lr_end_final,
                    label_smoothing=0.15,
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
                val_dataloaders=tmp_valid_loader,
            )
            evaluator.fit(model)
            model = model.to(self.device)

            # Сохранение модели
            save_path = Path(self.config.output_path) / "trained_models_pth"
            save_path.mkdir(parents=True, exist_ok=True)
            model_path = save_path / f"model_{model_id}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model {model_id} saved to {model_path}")

            return model

        except Exception as e:
            print(f"Error training model {model_id}: {e}")
            return None

    def evaluate_and_save_results(
        self,
        model: torch.nn.Module,
        architecture: dict,
        data_loader: DataLoader,
        folder_name: str,
        model_id: int,
        mode: str = "class",
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

    def collect_ensemble_stats(
        self, test_loader: DataLoader
    ) -> Optional[Dict[str, Any]]:
        """Собирает статистику для ансамбля: точность, ECE."""
        valid_models = [m for m in self.models if m is not None]
        if not valid_models:
            print("No valid models for ensemble evaluation.")
            return None

        main_device = self.device
        for model in valid_models:
            model.to(main_device).eval()

        total = 0
        correct_ensemble = 0
        correct_models = [0] * len(valid_models)

        n_bins = self.config.n_ece_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_conf_sums = torch.zeros(n_bins)
        bin_acc_sums = torch.zeros(n_bins)
        bin_counts = torch.zeros(n_bins)

        with torch.inference_mode():
            for images, labels in tqdm(test_loader, desc="Evaluating ensemble"):
                images, labels = images.to(main_device), labels.to(main_device)
                batch_size = labels.size(0)
                total += batch_size

                avg_output = None
                for idx, model in enumerate(valid_models):
                    output = model(images).softmax(dim=1)
                    _, preds = output.max(1)
                    correct_models[idx] += (preds == labels).sum().item()

                    if avg_output is None:
                        avg_output = torch.zeros_like(output)
                    avg_output += output

                avg_output /= len(valid_models)
                confidences, preds_ens = avg_output.max(1)
                correct_ens_batch = (preds_ens == labels).float()

                correct_ensemble += correct_ens_batch.sum().item()

                confidences = confidences.cpu().float()
                correct_ens_batch = correct_ens_batch.cpu()

                for conf, correct in zip(confidences, correct_ens_batch):
                    bin_idx = torch.bucketize(conf, bin_boundaries, right=True) - 1
                    bin_idx = bin_idx.clamp(min=0, max=n_bins - 1)
                    bin_counts[bin_idx] += 1
                    bin_conf_sums[bin_idx] += conf
                    bin_acc_sums[bin_idx] += correct

                if self.config.developer_mode:
                    break

        return {
            "total": total,
            "correct_ensemble": correct_ensemble,
            "correct_models": correct_models,
            "bin_counts": bin_counts,
            "bin_conf_sums": bin_conf_sums,
            "bin_acc_sums": bin_acc_sums,
            "n_bins": n_bins,
            "num_models": len(valid_models),
        }

    def finalize_ensemble_evaluation(
        self, stats: Optional[Dict[str, Any]], file_name: str = "ensemble_results"
    ) -> Tuple[Optional[float], Optional[List[float]], Optional[float]]:
        """Финализирует оценку ансамбля: вычисляет метрики и сохраняет результаты."""
        if stats is None:
            return None, None, None

        total = stats["total"]
        correct_ensemble = stats["correct_ensemble"]
        correct_models = stats["correct_models"]
        bin_counts = stats["bin_counts"]
        bin_conf_sums = stats["bin_conf_sums"]
        bin_acc_sums = stats["bin_acc_sums"]
        n_bins = stats["n_bins"]
        num_models = stats["num_models"]

        ensemble_acc = 100.0 * correct_ensemble / total
        model_accs = [100.0 * c / total for c in correct_models]

        ece = 0.0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_conf_avg = bin_conf_sums[i] / bin_counts[i]
                bin_acc_avg = bin_acc_sums[i] / bin_counts[i]
                bin_weight = bin_counts[i] / total
                ece += bin_weight * abs(bin_conf_avg - bin_acc_avg)

        print(f"\nEnsemble Evaluation Results:")
        print(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%")
        print(f"Ensemble ECE: {ece:.4f}")
        for i, acc in enumerate(model_accs):
            print(f"Model {i+1} Top-1 Accuracy: {acc:.2f}%")

        # Сохранение
        output_path = Path(self.config.output_path)
        output_path.mkdir(exist_ok=True)
        experiment_num = self._get_free_file_index(str(output_path), file_name)
        out_file = output_path / f"{file_name}_{experiment_num}.txt"

        with open(out_file, "w") as f:
            f.write(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%\n")
            f.write(f"Ensemble ECE: {ece:.4f}\n")
            f.write(f"Number of models: {num_models}\n")
            for i, acc in enumerate(model_accs):
                f.write(f"Model {i+1} Accuracy: {acc:.2f}%\n")

        print(f"Results saved to {out_file}")
        return ensemble_acc, model_accs, ece

    def _get_free_file_index(self, path: str, file_name: str) -> int:
        """Находит свободный индекс для имени файла."""
        experiment_num = 0
        while (Path(path) / f"{file_name}_{experiment_num}.txt").exists():
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
    ):
        """Целевой процесс для параллельного обучения."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("high")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            runner = DiversityNESRunner(
                config, DatasetsInfo.get(config.dataset_name.lower())
            )
            train_loader, valid_loader, test_loader = runner.get_data_loaders()

            model = runner.train_model(
                architecture, train_loader, valid_loader, model_id
            )
            if model is not None:
                model = model.to(device)

            eval_loader = valid_loader
            eval_folder = Path(config.output_path) / "trained_models_archs"
            runner.evaluate_and_save_results(
                model, architecture, eval_loader, str(eval_folder), model_id=model_id
            )

        except Exception as e:
            print(f"Error in process {model_id}: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def fill_models_list(self) -> None:
        """Загружает обученные модели из .pth и .json."""
        archs_path = Path(self.config.output_path) / "trained_models_archs"
        pth_path = Path(self.config.output_path) / "trained_models_pth"

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
        """Загружает и оценивает предобученные модели по индексу."""
        root_dir = Path(self.config.best_models_save_path)
        json_dir = root_dir / f"models_json_{index}"
        pth_dir = root_dir / f"models_pth_{index}"

        if not json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")
        if not pth_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pth_dir}")

        print(f"Using JSON from: {json_dir}, PTH from: {pth_dir}")
        arch_dicts = load_json_from_directory(json_dir)
        if not arch_dicts:
            raise RuntimeError("No architectures found")

        _, valid_loader, test_loader = self.get_data_loaders()

        for entry in tqdm(arch_dicts, desc=f"Evaluating models index {index}"):
            arch, model_id = entry["architecture"], entry.get("id")
            if model_id is None:
                raise ValueError("Missing 'id' in architecture")

            with model_context(arch):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=self.dataset_key,
                )
            model.to(self.device)
            pth_file = pth_dir / f"model_{model_id}.pth"
            model.load_state_dict(
                torch.load(pth_file, map_location=self.device, weights_only=True)
            )
            self.models.append(model)

            if not self.config.evaluate_ensemble_flag:
                self.evaluate_and_save_results(
                    model,
                    arch,
                    valid_loader,
                    folder_name=str(
                        Path(self.config.output_path) / f"trained_models_archs_{index}"
                    ),
                    model_id=model_id,
                )

        if self.config.evaluate_ensemble_flag:
            stats = self.collect_ensemble_stats(test_loader)
            self.finalize_ensemble_evaluation(stats, f"ensemble_results_{index}")

        print(f"✅ Pretrained models from index {index} evaluated.")

    def run_all_pretrained(self) -> None:
        """Оценивает все сохранённые ансамбли."""
        root_dir = Path(self.config.best_models_save_path)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        last_index = self.get_latest_index_from_dir(root_dir)
        for idx in range(1, last_index + 1):
            if (root_dir / f"models_json_{idx}").exists() and (
                root_dir / f"models_pth_{idx}"
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
        if self.config.evaluate_ensemble_flag:
            latest_index = self.get_latest_index_from_dir()
            models_json_dir = (
                Path(self.config.best_models_save_path) / f"models_json_{latest_index}"
            )
            arch_dicts = load_json_from_directory(models_json_dir)
        else:
            arch_dicts = generate_arch_dicts(self.config.n_models_to_evaluate)

        archs = [d["architecture"] for d in arch_dicts]
        n_models = len(archs)

        available_gpus = self._get_available_gpus()
        n_gpus = len(available_gpus)
        max_per_gpu = self.config.max_per_gpu
        total_processes = n_models

        print(f"Available GPUs: {available_gpus}")
        print(f"Training {n_models} models, up to {max_per_gpu} per GPU")

        # Создание директорий
        output_path = Path(self.config.output_path)
        (output_path / "trained_models_pth").mkdir(parents=True, exist_ok=True)
        (output_path / "trained_models_archs").mkdir(parents=True, exist_ok=True)

        # Параллельное обучение
        processes = []
        for idx, arch in enumerate(archs):
            gpu_idx = (idx // max_per_gpu) % n_gpus
            physical_gpu_id = available_gpus[gpu_idx]

            p = mp.get_context("spawn").Process(
                target=self.train_single_model_process,
                args=(
                    arch,
                    idx,
                    physical_gpu_id,
                    self.config,
                    self.dataset_key,
                    self.num_classes,
                ),
            )
            p.start()
            processes.append(p)
            print(f"Started model {idx} on GPU {physical_gpu_id}")

            while len(processes) >= n_gpus * max_per_gpu:
                time.sleep(2)
                for p in processes[:]:
                    if not p.is_alive():
                        p.join()
                        processes.remove(p)
                        break

        for p in processes:
            p.join()
        print("✅ All models trained!")

        # Оценка ансамбля
        if self.config.evaluate_ensemble_flag:
            print("📊 Evaluating ensemble...")
            self.fill_models_list()
            if self.models:
                _, _, test_loader = self.get_data_loaders()
                stats = self.collect_ensemble_stats(test_loader)
                self.finalize_ensemble_evaluation(stats, "ensemble_results")
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

    config = TrainConfig(**params)
    info = DatasetsInfo.get(config.dataset_name.lower())
    runner = DiversityNESRunner(config, info)

    if config.use_pretrained_models_for_ensemble:
        runner.run_all_pretrained()
    else:
        runner.run()
