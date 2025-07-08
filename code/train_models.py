import os
import json
import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import nni
from torch.utils.data import SubsetRandomSampler, SequentialSampler, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from nni.nas.evaluator.pytorch import DataLoader, Lightning, Trainer
from nni.nas.space import model_context
from tqdm import tqdm

from DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.data_generator import generate_arch_dicts
from dependencies.darts_classification_module import DartsClassificationModule
from dependencies.train_config import TrainConfig


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
        # Set random seeds for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Determine dataset specifics
        if config.dataset_name.lower() == "cifar10":
            self.MEAN = [0.4914, 0.4822, 0.4465]
            self.STD = [0.2470, 0.2435, 0.2616]
            self.img_size = 32
            self.base_transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        elif config.dataset_name.lower() == "cifar100":
            self.MEAN = [0.5071, 0.4867, 0.4408]
            self.STD = [0.2673, 0.2564, 0.2762]
            self.img_size = 32
            self.base_transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        elif config.dataset_name.lower() == "fashionmnist":
            self.MEAN = [0.2860406]
            self.STD = [0.35302424]
            self.img_size = 32
            self.base_transform = [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
            ]
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

    def get_data_loaders(self, batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders for the chosen dataset.
        """
        bs = batch_size or self.config.batch_size
        transform = transforms.Compose(
            self.base_transform
            + [
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

        if self.config.dataset_name.lower() == "cifar10":
            dataset_cls = CIFAR10
        elif self.config.dataset_name.lower() == "cifar100":
            dataset_cls = CIFAR100
        elif self.config.dataset_name.lower() == "fashionmnist":
            dataset_cls = FashionMNIST

        train_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=True,
            download=True,
            transform=transform,
        )
        num_samples = len(train_data)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        split = int(num_samples * self.config.train_size)

        train_subset = Subset(train_data, indices[:split])
        valid_subset = Subset(train_data, indices[split:])

        train_loader = DataLoader(
            train_subset,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_subset,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return train_loader, valid_loader

    def train_model(
        self,
        architecture: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_id: int,
    ) -> torch.nn.Module:
        """
        Train a single model defined by architecture and return the trained model.
        """
        try:
            with model_context(architecture):
                if self.config.dataset_name.lower() == "fashionmnist":
                    model = DartsSpace(
                        width=self.config.width,
                        num_cells=self.config.num_cells,
                        dataset="fashionmnist",
                    )
                else:
                    dataset_type = (
                        "cifar"
                        if self.config.dataset_name.lower() == "cifar10"
                        else "cifar100"
                    )
                    model = DartsSpace(
                        width=self.config.width,
                        num_cells=self.config.num_cells,
                        dataset=dataset_type,
                    )

            model.to(self.device)

            devices_arg = 1 if self.device.type == "cuda" else "auto"
            accelerator = "gpu" if self.device.type == "cuda" else "cpu"

            evaluator = Lightning(
                DartsClassificationModule(
                    learning_rate=self.config.lr_final,
                    weight_decay=3e-4,
                    auxiliary_loss_weight=0.4,
                    max_epochs=self.config.n_epochs_final,
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
            return model
        except Exception as e:
            print(f"Error training model {model_id}: {str(e)}")
            return None

    def evaluate_and_save(
        self,
        model: torch.nn.Module,
        architecture: dict,
        model_id: int,
        valid_loader: DataLoader,
    ):
        """
        Evaluate the model's performance on validation data and save results.
        """
        if model is None:
            print(f"Skipping evaluation for failed model {model_id}")
            return

        out_folder = self.config.output_path
        os.makedirs(out_folder, exist_ok=True)
        model.to(self.device).eval()

        total, correct = 0, 0
        predictions = []
        with torch.inference_mode():  # More efficient than no_grad
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = torch.softmax(model(imgs), dim=1)
                predictions.extend(outputs.cpu().tolist())
                _, preds = outputs.max(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        result = {
            "model_id": model_id,
            "architecture": architecture,
            "valid_accuracy": acc,
        }

        file_path = os.path.join(out_folder, f"model_{model_id:04d}_results.json")
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved results for model {model_id} to {file_path}")

    def evaluate_ensemble(self, valid_loader):
        """
        Evaluate ensemble of models: compute Top-1 accuracy and Expected Calibration Error (ECE),
        save and print results.
        """
        if not hasattr(self, 'models') or not self.models:
            print("No models available for ensemble evaluation.")
            return None, None, None

        main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Filter out None models (failed training)
        valid_models = [m for m in self.models if m is not None]
        
        if not valid_models:
            print("No valid models for ensemble evaluation.")
            return None, None, None
            
        # Move models to device and set to eval mode
        for i, model in enumerate(valid_models):
            valid_models[i] = model.to(main_device).eval()
        
        total = 0
        correct_ensemble = 0
        correct_models = [0] * len(valid_models)
        
        # For ECE calculation
        n_bins = self.config.n_ece_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_conf_sums = torch.zeros(n_bins)
        bin_acc_sums = torch.zeros(n_bins)
        bin_counts = torch.zeros(n_bins)

        with torch.inference_mode():  # More efficient than no_grad
            for images, labels in tqdm(valid_loader, desc="Evaluating ensemble"):
                images, labels = images.to(main_device), labels.to(main_device)
                batch_size = labels.size(0)
                total += batch_size
                
                # Initialize average output
                avg_output = None
                
                for idx, model in enumerate(valid_models):
                    output = model(images)
                    
                    # Calculate individual model accuracy
                    probs = output.softmax(dim=1)
                    _, preds = probs.max(1)
                    correct_models[idx] += preds.eq(labels).sum().item()
                    
                    # Accumulate outputs for ensemble average
                    if avg_output is None:
                        avg_output = torch.zeros_like(output)
                    avg_output += output
                
                # Compute ensemble predictions
                avg_output /= len(valid_models)
                ensemble_probs = avg_output.softmax(dim=1)
                confidences, preds_ens = ensemble_probs.max(1)
                correct_ens_batch = preds_ens.eq(labels)
                correct_ensemble += correct_ens_batch.sum().item()
                
                # Update ECE bins
                confidences = confidences.cpu().float()
                correct_ens_batch = correct_ens_batch.cpu().float()
                
                for conf, correct in zip(confidences, correct_ens_batch):
                    bin_idx = torch.bucketize(conf, bin_boundaries, right=True) - 1
                    if bin_idx < 0:  # Handle case when confidence is exactly 0.0
                        bin_idx = 0
                    bin_counts[bin_idx] += 1
                    bin_conf_sums[bin_idx] += conf
                    bin_acc_sums[bin_idx] += correct

        # Calculate metrics
        ensemble_acc = 100.0 * correct_ensemble / total
        model_accs = [100.0 * c / total for c in correct_models]
        
        # Calculate ECE
        ece = 0.0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_conf_avg = bin_conf_sums[i] / bin_counts[i]
                bin_acc_avg = bin_acc_sums[i] / bin_counts[i]
                bin_weight = bin_counts[i] / total
                ece += bin_weight * abs(bin_conf_avg - bin_acc_avg)

        # Print and save results
        print(f"\nEnsemble Evaluation Results:")
        print(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%")
        print(f"Ensemble ECE: {ece:.4f}")
        for i, acc in enumerate(model_accs):
            print(f"Model {i+1} Top-1 Accuracy: {acc:.2f}%")
        
        os.makedirs(self.config.output_path, exist_ok=True)
        out_file = os.path.join(self.config.output_path, "ensemble_results.txt")
        with open(out_file, "w") as f:
            f.write(f"Ensemble Top-1 Accuracy: {ensemble_acc:.2f}%\n")
            f.write(f"Ensemble ECE: {ece:.4f}\n")
            f.write(f"Number of models: {len(valid_models)}\n")
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

        os.makedirs(self.config.output_path, exist_ok=True)

        print("Loading architecture definitions...")
        arch_dicts = load_json_from_directory(self.config.best_models_save_path)
        if not arch_dicts:
            raise RuntimeError(
                "No valid architectures found in the specified directory"
            )

        print("Creating data loaders...")
        train_loader, valid_loader = self.get_data_loaders()

        print(f"Starting training for {len(arch_dicts)} models...")
        archs = [d["architecture"] for d in arch_dicts]

        for idx, arch in enumerate(tqdm(archs, desc="Training models")):
            print(f"\nTraining model {idx+1}/{len(archs)}")
            model = self.train_model(arch, train_loader, valid_loader, idx)
            self.evaluate_and_save(model, arch, idx, valid_loader)

        print("\nEvaluating ensemble...")
        self.evaluate_ensemble(valid_loader)

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
