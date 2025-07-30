import os
import json
import shutil
import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import nni
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from nni.nas.evaluator.pytorch import DataLoader, Lightning, Trainer
from nni.nas.space import model_context
from tqdm import tqdm

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
                        json_data.append(json.load(f))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
    return json_data


class DiversityNESRunner:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.models = []
        self.model_id = 0

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Determine dataset specifics
        if config.dataset_name.lower() == "cifar10":
            self.MEAN = [0.4914, 0.4822, 0.4465]
            self.STD = [0.2470, 0.2435, 0.2616]
            self.img_size = 32
            self.base_transform = []
        elif config.dataset_name.lower() == "cifar100":
            self.MEAN = [0.5071, 0.4867, 0.4408]
            self.STD = [0.2673, 0.2564, 0.2762]
            self.img_size = 32
            self.base_transform = []
        elif config.dataset_name.lower() == "fashionmnist":
            self.MEAN = [0.2860406]
            self.STD = [0.35302424]
            self.img_size = 32
            self.base_transform = [
                transforms.Resize(32),
            ]
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

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
        train_transform = transforms.Compose(
            self.base_transform
            + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )
        test_transform = transforms.Compose(
            self.base_transform
            + [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

        if self.config.dataset_name.lower() == "cifar10":
            dataset_cls = CIFAR10
            self.num_classes = 10
        elif self.config.dataset_name.lower() == "cifar100":
            dataset_cls = CIFAR100
            self.num_classes = 100
        elif self.config.dataset_name.lower() == "fashionmnist":
            dataset_cls = FashionMNIST
            self.num_classes = 10

        train_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=False,
            download=True,
            transform=test_transform,
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
                num_workers=self.config.num_workers,
                shuffle=False,
            )

        train_subset = Subset(train_data, indices[:split])
        train_loader = DataLoader(
            train_subset,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=bs,
            num_workers=self.config.num_workers,
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
        self,
        architecture: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_id: int,
        save_dir_name="trained_models",
        weight_init_seed=None,
    ) -> torch.nn.Module:
        """
        Train a single model defined by architecture and return the trained model.
        """
        try:
            if self.config.dataset_name.lower() == "cifar10":
                dataset = "cifar"
            elif self.config.dataset_name.lower() == "cifar100":
                dataset = "cifar100"
            elif self.config.dataset_name.lower() == "fashionmnist":
                dataset = "cifar"
            else:
                raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

            with model_context(architecture):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=dataset,
                )

            if weight_init_seed:
                torch.manual_seed(weight_init_seed)
                model.apply(self._custom_weight_init)
            model.to(self.device)

            devices_arg = 1 if self.device.type == "cuda" else "auto"
            accelerator = "gpu" if self.device.type == "cuda" else "cpu"

            evaluator = Lightning(
                DartsClassificationModule(
                    learning_rate=self.config.lr_start_final,
                    weight_decay=3e-4,
                    auxiliary_loss_weight=0.4,
                    max_epochs=self.config.n_epochs_final,
                    num_classes=self.num_classes,
                    lr_final=self.config.lr_end_final,
                ),
                trainer=Trainer(
                    gradient_clip_val=5.0,
                    max_epochs=self.config.n_epochs_final,
                    fast_dev_run=self.config.developer_mode,
                    accelerator=accelerator,
                    devices=devices_arg,
                    enable_progress_bar=True,
                ),
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )
            evaluator.fit(model)
            self.models.append(model)

            os.makedirs(Path(self.config.output_path) / save_dir_name, exist_ok=True)
            print(
                f"Saving model {model_id} to {self.config.output_path} {save_dir_name}"
            )
            torch.save(
                model.state_dict(),
                Path(self.config.output_path) / save_dir_name / f"model_{model_id}.pth",
            )

            return model

        except Exception as e:
            print(f"Error training model {model_id}: {str(e)}")
            return None

    def evaluate_and_save_results(
        self,
        model,
        architecture,
        valid_loader,
        folder_name="results_dataset",
    ):
        """
        Оценивает модель на валидационном наборе данных и сохраняет результаты в JSON.
        Аргументы:
        model: Обученная модель
        architecture: Архитектура модели
        valid_loader (DataLoader): DataLoader для валидационных данных
        model_id: Уникальный идентификатор модели
        folder_name (str): Папка для сохранения результатов
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(folder_name, exist_ok=True)

        # Перенос модели на устройство и режим оценки
        model.to(device)
        model.eval()

        valid_correct = 0
        valid_total = 0
        valid_preds = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                valid_preds.extend(outputs.cpu().tolist())
                _, predicted = torch.max(outputs, 1)
                valid_correct += (predicted == labels).sum().item()
                valid_total += labels.size(0)

                if self.config.developer_mode:
                    break

        valid_accuracy = valid_correct / valid_total

        # Формирование результата
        result = {
            "architecture": architecture,
            "valid_predictions": valid_preds,
            "valid_accuracy": valid_accuracy,
        }

        # Генерация имени файла с использованием model_id
        file_name = f"model_{self.model_id:d}_results.json"
        file_path = os.path.join(folder_name, file_name)

        # Сохранение результатов
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Results for model_{self.model_id} saved to {file_path}")
        self.model_id += 1

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

    def run(self):
        """
        Main pipeline: load architectures, load data, train and evaluate all models.
        """
        if not Path(self.config.best_models_save_path).exists():
            raise FileNotFoundError(
                f"Architecture directory {self.config.best_models_save_path} not found"
            )

        if self.config.evaluate_ensemble_flag:
            print("Loading architecture definitions...")
            arch_dicts = load_json_from_directory(self.config.best_models_save_path)
            if not arch_dicts:
                raise RuntimeError(
                    "No valid architectures found in the specified directory"
                )
        else:
            arch_dicts = generate_arch_dicts(self.config.n_models_to_evaluate)

        print("Creating data loaders...")
        train_loader, valid_loader, test_loader = self.get_data_loaders()

        print(f"Starting training for {len(arch_dicts)} models...")
        archs = [d["architecture"] for d in arch_dicts]

        for idx, arch in enumerate(tqdm(archs, desc="Training models")):
            print(f"\nTraining model {idx+1}/{len(archs)}")
            model = self.train_model(arch, train_loader, valid_loader, idx)
            self.evaluate_and_save_results(
                model, arch, valid_loader, self.config.output_path + "/trained_models_archs/"
            )
            if not self.config.evaluate_ensemble_flag:
                self.models = []  # Doesn't need to save models in prepare dataset mode

        if self.config.evaluate_ensemble_flag:
            print("\nEvaluating ensemble...")
            stats = self.collect_ensemble_stats(test_loader)
            self.finalize_ensemble_evaluation(stats, "ensemble_results")

        print("\nAll models processed successfully!")


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

    runner = DiversityNESRunner(config)
    runner.run()
