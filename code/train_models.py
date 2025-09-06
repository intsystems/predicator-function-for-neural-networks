import os
import json
import shutil
import re
import argparse
from pathlib import Path
import random
from typing import List, Tuple
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

from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.darts_classification_module import DartsClassificationModule
from dependencies.train_config import TrainConfig
from dependencies.data_generator import generate_arch_dicts


def load_json_from_directory(directory_path: str) -> List[dict]:
    """
    Load all JSON files from a directory into a list of dicts.
    """
    if not Path(directory_path).exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    json_data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tmp_data = json.load(f)
                        m = re.search(r"(\d+)\.json$", file)
                        if m and "id" not in tmp_data:
                            tmp_data["id"] = int(m.group(1))
                        json_data.append(tmp_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
    return json_data

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
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
        
        mask[y1:y2, x1:x2] = 0.
        mask = mask.unsqueeze(0).expand_as(img)
        
        return img * mask



class DatasetsInfo(object):
    DATASETS = {
        "cifar10": {
            "key": "cifar",
            "class": CIFAR10,
            "num_classes": 10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "img_size": 32,
            "train_transform": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                Cutout(16),
            ]),
            "test_transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ]),
        },
        "cifar100": {
            "key": "cifar100",
            "class": CIFAR100,
            "num_classes": 100,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2673, 0.2564, 0.2762],
            "img_size": 32,
            "train_transform": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]),
                Cutout(16),
            ]),
            "test_transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]),
            ]),
        },
        "fashionmnist": {
            "key": "cifar",
            "class": FashionMNIST,
            "num_classes": 10,
            "mean": [0.2860406],
            "std": [0.35302424],
            "img_size": 32,
            "train_transform": [transforms.Resize(32)],
            "test_transform": [transforms.Resize(32)],
        },
    }

    @classmethod
    def get(cls, dataset_name: str):
        """Возвращает конфигурацию датасета по его имени.
        
        Args:
            dataset_name: Имя датасета (например, 'cifar10', 'cifar100', 'fashionmnist')
            
        Returns:
            Словарь с конфигурацией датасета
            
        Raises:
            ValueError: Если датасет с указанным именем не найден
        """
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(cls.DATASETS.keys())}")
        return cls.DATASETS[dataset_name]

class DiversityNESRunner:
    def __init__(self, config: TrainConfig, info: DatasetsInfo):
        self.config = config
        self.models = []
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
        self, batch_size: int = None, seed=None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create training and validation data loaders for the chosen dataset.
        """
        bs = batch_size if batch_size else self.config.batch_size_final

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
        np_seed = seed if seed else self.config.seed
        np.random.seed(np_seed)
        np.random.shuffle(indices)

        if self.config.evaluate_ensemble_flag:
            split = num_samples
            valid_subset = None
            valid_loader = None
        else:
            split = int(num_samples * self.config.train_size)
            valid_subset = Subset(train_data, indices[split:])
            valid_loader = DataLoader(
                valid_subset,
                batch_size=bs,
                num_workers=0,
                shuffle=False,
            )

        train_subset = Subset(train_data, indices[:split])
        train_loader = DataLoader(
            train_subset,
            batch_size=bs,
            num_workers=0,
            shuffle=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=bs,
            num_workers=0,
            shuffle=False,
        )

        return train_loader, valid_loader, test_loader

    @staticmethod
    def _custom_weight_init(module):
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def train_model(
        self, architecture: dict, train_loader, valid_loader, model_id
    ) -> torch.nn.Module:
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
                    weight_decay=3e-4,
                    auxiliary_loss_weight=0.4,
                    max_epochs=self.config.n_epochs_final,
                    num_classes=self.num_classes,
                    lr_final=self.config.lr_end_final,
                    label_smoothing=0.15
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

            # === Гарантированно создаём папку (на всякий случай) ===
            save_path = Path(self.config.output_path) / "trained_models_pth"
            save_path.mkdir(parents=True, exist_ok=True)

            model_path = save_path / f"model_{model_id}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model {model_id} saved to {model_path}")

            return model

        except Exception as e:
            print(f"Error training model {model_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def evaluate_and_save_results(
        self,
        model,
        architecture,
        valid_loader,
        folder_name,
        mode="class",
        model_id=None,
    ):
        device = next(model.parameters()).device
        model.eval()

        valid_correct = 0
        valid_total = 0
        valid_preds = []

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                valid_correct += (predicted == labels).sum().item()
                valid_total += labels.size(0)
                if mode == "logits":
                    valid_preds.extend(outputs.cpu().tolist())
                else:
                    valid_preds.extend(predicted.cpu().tolist())

        valid_accuracy = valid_correct / valid_total

        result = {
            "architecture": architecture,
            "valid_predictions": valid_preds,
            "valid_accuracy": valid_accuracy,
        }

        file_name = f"model_{model_id}.json"
        file_path = os.path.join(folder_name, file_name)

        os.makedirs(folder_name, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Results for model_{model_id} saved to {file_path}")

    def collect_ensemble_stats(self, test_loader):
        if not hasattr(self, "models") or not self.models:
            print("No models available for ensemble evaluation.")
            return None

        main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        valid_models = [m for m in self.models if m is not None]

        if not valid_models:
            print("No valid models for ensemble evaluation.")
            return None

        for i, model in enumerate(valid_models):
            valid_models[i] = model.to(main_device).eval()

        total = 0
        correct_ensemble = 0
        correct_models = [0] * len(valid_models)

        # For ECE
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
                    output = model(images)
                    probs = output.softmax(dim=1)
                    _, preds = probs.max(1)
                    correct_models[idx] += (preds == labels).sum().item()

                    if avg_output is None:
                        avg_output = torch.zeros_like(probs)
                    avg_output += probs

                avg_output /= len(valid_models)
                confidences, preds_ens = avg_output.max(1)
                correct_ens_batch = preds_ens == labels
                correct_ensemble += correct_ens_batch.sum().item()

                confidences = confidences.cpu().float()
                correct_ens_batch = correct_ens_batch.cpu().float()

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

    def _get_free_file_index(self, path: Path, file_name="ensembles_results"):
        experiment_num = 0
        while os.path.exists(path + f"/{file_name}_{experiment_num}.txt"):
            experiment_num += 1
        return experiment_num

    def finalize_ensemble_evaluation(self, stats, file_name="ensembles_results"):
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

        # Print results
        print(f"\nEnsemble Evaluation Results:")
        print(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%")
        print(f"Ensemble ECE: {ece:.4f}")
        for i, acc in enumerate(model_accs):
            print(f"Model {i+1} Top-1 Accuracy: {acc:.2f}%")

        # Save to file
        os.makedirs(self.config.output_path, exist_ok=True)
        experiment_num = self._get_free_file_index(self.config.output_path, file_name)
        out_file = os.path.join(
            self.config.output_path, f"{file_name}_{experiment_num}.txt"
        )
        with open(out_file, "w") as f:
            f.write(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%\n")
            f.write(f"Ensemble ECE: {ece:.4f}\n")
            f.write(f"Number of models: {num_models}\n")
            for i, acc in enumerate(model_accs):
                f.write(f"Model {i+1} Accuracy: {acc:.2f}%\n")

        print(f"Results saved to {out_file}")
        return ensemble_acc, model_accs, ece

    def get_latest_index_from_dir(self, base_path: Path = None) -> int:
        """
        Находит максимальный индекс среди директорий models_json_*
        """
        if base_path is None:
            base_path = Path(self.config.best_models_save_path)

        candidates = [p for p in base_path.glob("models_json_*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No models_json_* directories found in {base_path}"
            )

        indices = [
            int(m.group(1))
            for p in candidates
            if (m := re.search(r"models_json_(\d+)", str(p)))
        ]

        if not indices:
            raise ValueError("No valid indices found in models_json_* directories")

        return max(indices)

    @staticmethod
    def train_single_model_process(
        architecture, model_id, physical_gpu_id, config, dataset_key, num_classes
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        try:
            # === ОЧИСТКА ПАМЯТИ GPU ПЕРЕД НАЧАЛОМ ===
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize(0)
            # ===============================

            torch.set_float32_matmul_precision("high")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            runner = DiversityNESRunner(
                config, DatasetsInfo.get(config.dataset_name.lower())
            )
            train_loader, valid_loader, test_loader = runner.get_data_loaders()

            if valid_loader is None:
                model = runner.train_model(
                    architecture, train_loader, test_loader, model_id
                )
            else:
                model = runner.train_model(
                    architecture, train_loader, valid_loader, model_id
                )

            if model is not None:
                model = model.to(device)

            if valid_loader is None:
                runner.evaluate_and_save_results(
                    model,
                    architecture,
                    test_loader,
                    folder_name=config.output_path + "/trained_models_archs/",
                    model_id=model_id,
                )
            else:
                runner.evaluate_and_save_results(
                    model,
                    architecture,
                    valid_loader,
                    folder_name=config.output_path + "/trained_models_archs/",
                    model_id=model_id,
                )

        except Exception as e:
            print(f"Error in process {model_id}: {e}")
            raise

        finally:
            # === ФИНАЛЬНАЯ ОЧИСТКА ПАМЯТИ ===
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
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

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",")]
        else:
            available_gpus = list(range(torch.cuda.device_count()))

        n_gpus = len(available_gpus)
        print(f"Available physical GPUs: {available_gpus}")
        print(f"Will train {n_models} models using up to {n_gpus} GPUs in parallel.")

        output_path = Path(self.config.output_path)
        (output_path / "trained_models_pth").mkdir(parents=True, exist_ok=True)
        (output_path / "trained_models_archs").mkdir(parents=True, exist_ok=True)

        print(f"Created output directories in {output_path}")
        print("Starting training...")

        # === ЗАПУСК С ОГРАНИЧЕНИЕМ: НЕ БОЛЕЕ n_gpus ПРОЦЕССОВ ОДНОВРЕМЕННО ===
        processes = []
        active_processes = {}

        for idx, arch in enumerate(archs):
            physical_gpu_id = available_gpus[idx % n_gpus]

            while len(processes) >= n_gpus:
                for p_idx, p in list(active_processes.items()):
                    if not p.is_alive():
                        p.join()
                        processes.remove(p)
                        del active_processes[p_idx]
                        break
                time.sleep(0.5)

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
            active_processes[idx] = p

            print(f"Started process {idx} on GPU {physical_gpu_id}")

        for p in processes:
            p.join()

        print("All models trained! Logs saved in 'logs/' directory.")

        if self.config.evaluate_ensemble_flag:
            self.fill_models_list()
            _, _, test_loader = runner.get_data_loaders()
            root_dir = Path(self.config.best_models_save_path)
            last_index = self.get_latest_index_from_dir(root_dir)
            print(f"\nEvaluating ensemble at index {last_index}...")
            stats = self.collect_ensemble_stats(test_loader)
            self.finalize_ensemble_evaluation(stats, f"ensemble_results_{last_index}")

        print(
            f"\nAll pretrained models from index {last_index} evaluated successfully!"
        )

    def fill_models_list(self):
        archs_path = Path(self.config.output_path) / "trained_models_archs"
        pth_path = Path(self.config.output_path) / "trained_models_pth"

        if not archs_path.exists():
            raise FileNotFoundError(f"Directory not found: {archs_path}")
        if not pth_path.exists():
            raise FileNotFoundError(f"Directory not found: {pth_path}")

        self.models = []
        device = self.device

        json_files = sorted(archs_path.glob("model_*.json"))
        if not json_files:
            print("No .json files found in trained_models_archs/")
            return

        print(f"Found {len(json_files)} models to load. Loading...")

        for json_file in json_files:
            try:
                m = re.search(r"model_(\d+)\.json", json_file.name)
                if not m:
                    print(f"Skipping invalid filename: {json_file.name}")
                    continue
                model_id = int(m.group(1))

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
                    print(f"Weights not found for model_{model_id} at {pth_file}")
                    continue

                state_dict = torch.load(pth_file, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model = model.to(device)
                self.models.append(model)

                print(f"✅ Model {model_id} loaded and added to self.models")

            except Exception as e:
                print(f"❌ Failed to load model from {json_file}: {e}")
                import traceback

                traceback.print_exc()

        print(f"Successfully loaded {len(self.models)} models into self.models.")

    def run_all_pretrained(self):
        """
        Wrapper over run_pretrained(index) that determines whether to evaluate all saved models
        or just the latest one based on config.evaluate_all_pretrained_models.
        """
        root_dir = Path(self.config.best_models_save_path)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory {root_dir} not found")
        last_index = self.get_latest_index_from_dir(root_dir)

        for idx in range(1, last_index + 1):
            json_dir = root_dir / f"models_json_{idx}"
            pth_dir = root_dir / f"models_pth_{idx}"
            if json_dir.exists() and pth_dir.exists():
                self.run_pretrained(idx)
                self.models = []
            else:
                print(f"Skipping index {idx}: missing json or pth directory.")

    def _extract_index(self, path):
        match = re.search(r"models_json_(\d+)", str(path))
        return int(match.group(1)) if match else -1

    def run_pretrained(self, index: int):
        """
        Load pretrained models and evaluate them without further training
        using models_json_{index} and models_pth_{index}.
        """
        root_dir = Path(self.config.best_models_save_path)
        json_dir = root_dir / f"models_json_{index}"
        pth_dir = root_dir / f"models_pth_{index}"

        if not json_dir.exists():
            raise FileNotFoundError(f"Directory {json_dir} not found")
        if not pth_dir.exists():
            raise FileNotFoundError(f"Directory {pth_dir} not found")

        print(f"\nUsing models_json from: {json_dir}")
        print(f"Using models_pth from: {pth_dir}")

        arch_dicts = load_json_from_directory(json_dir)
        if not arch_dicts:
            raise RuntimeError(f"No valid architectures found in {json_dir}")

        _, valid_loader, test_loader = self.get_data_loaders()

        print(f"Loading and evaluating {len(arch_dicts)} pretrained models...")
        for idx, arch_entry in enumerate(
            tqdm(arch_dicts, desc=f"Evaluating models in index {index}")
        ):
            arch = arch_entry["architecture"]
            model_id = arch_entry.get("id")
            self.model_id = model_id
            if model_id is None:
                raise ValueError(f"Architecture entry {idx} has no 'id' field")

            dataset = self.dataset_key
            with model_context(arch):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=dataset,
                )

            model.to(self.device)

            model_path = pth_dir / f"model_{model_id}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.models.append(model)

            if not self.config.evaluate_ensemble_flag:
                self.evaluate_and_save_results(
                    model,
                    arch,
                    valid_loader,
                    self.config.output_path + f"/trained_models_archs_{index}/",
                    mode="class",
                )

        if self.config.evaluate_ensemble_flag:
            print(f"\nEvaluating ensemble at index {index}...")
            stats = self.collect_ensemble_stats(test_loader)
            self.finalize_ensemble_evaluation(stats, f"ensemble_results_{index}")

        print(f"\nAll pretrained models from index {index} evaluated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTS NAS Runner")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        required=True,
        help="Path to JSON file with TrainConfig parameters",
    )
    args = parser.parse_args()

    # Load hyperparameters from JSON and set device automatically
    params = json.loads(Path(args.hyperparameters_json).read_text())
    params.update({"device": "cuda:0" if torch.cuda.is_available() else "cpu"})

    config = TrainConfig(**params)
    info = DatasetsInfo.get(config.dataset_name.lower())
    runner = DiversityNESRunner(config, info)
    if config.use_pretrained_models_for_ensemble:
        runner.run_all_pretrained()
    else:
        runner.run()
