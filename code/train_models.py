import os
import json
import re
import argparse
from pathlib import Path
import random
from typing import List, Tuple, Optional, Dict, Any
import time
import copy

import numpy as np
import torch
import nni
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from torch.utils.data import Subset
from nni.nas.evaluator.pytorch import DataLoader
from nni.nas.space import model_context
from tqdm import tqdm

import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.dataset_info import DatasetsInfo
from dependencies.train_config import TrainConfig
from dependencies.train_utils import (
    load_json_from_directory,
    build_optimizer_and_scheduler,
    save_training_checkpoint,
    load_training_checkpoint,
)
from dependencies.metrics import (
    collect_ensemble_stats,
    calculate_nll,
    calculate_ece,
    calculate_oracle_nll,
    calculate_brier_score,
    MetricsCallback,
)

# === MAIN CLASS ===


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

    def prepare_datasets(self, download: bool = False):
        """Creates dataset objects without rebuilding split logic."""
        dataset_cls = self.dataset_cls
        train_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=True,
            download=download,
            transform=self.train_transform,
        )
        test_data = nni.trace(dataset_cls)(
            root=self.config.final_dataset_path,
            train=False,
            download=download,
            transform=self.test_transform,
        )
        return train_data, test_data

    def build_split_indices(self, num_samples: int, seed: Optional[int] = None):
        """Builds deterministic train/validation split indices once."""
        indices = list(range(num_samples))
        rng = np.random.default_rng(seed or self.config.seed)
        rng.shuffle(indices)
        split = int(num_samples * self.config.train_size_final)
        return indices[:split], indices[split:]

    def get_data_loaders(
        self,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        train_indices: Optional[List[int]] = None,
        valid_indices: Optional[List[int]] = None,
        download: bool = False,
    ):
        """Creates data loaders."""
        bs = batch_size or self.config.batch_size_final
        train_data, test_data = self.prepare_datasets(download=download)

        if train_indices is None or valid_indices is None:
            train_indices, valid_indices = self.build_split_indices(
                len(train_data), seed=seed
            )

        train_subset = Subset(train_data, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        valid_subset = Subset(train_data, valid_indices)
        valid_loader = DataLoader(
            valid_subset,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=bs,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

        print(
            f"[DATALOADER] pid={os.getpid()} bs={bs} num_workers={self.config.num_workers} "
            f"train_len={len(train_subset)} valid_len={len(valid_subset)} test_len={len(test_data)}"
        )
        return train_loader, valid_loader, test_loader

    @staticmethod
    def _custom_weight_init(module: torch.nn.Module) -> None:
        """Initializing weights for linear and convolutional layers."""
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
        checkpoint_path: Optional[Path] = None,
        resume_from_checkpoint: bool = False,
    ) -> Optional[torch.nn.Module]:
        """Trains one model in architecture."""
        seed = self.config.seed + model_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        metrics_callback = MetricsCallback()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        resume_state = None

        if resume_from_checkpoint and checkpoint_path is not None and checkpoint_path.exists():
            resume_state = load_training_checkpoint(checkpoint_path, map_location="cpu")
            print(
                f"[CHECKPOINT] Resuming model {model_id} from {checkpoint_path} "
                f"starting at epoch {resume_state.get('epoch', 0)}"
            )

        print(
            f"[TRAIN START] model_id={model_id} pid={os.getpid()} device={self.device} "
            f"train_loader_workers={getattr(train_loader, 'num_workers', 'n/a')} "
            f"valid_loader_is_none={valid_loader is None}"
        )

        try:
            with model_context(architecture):
                model = DartsSpace(
                    width=self.config.width,
                    num_cells=self.config.num_cells,
                    dataset=self.dataset_key,
                )
            model = model.to(self.device)
            model.apply(self._custom_weight_init)
            if resume_state is not None:
                model.load_state_dict(resume_state["model_state_dict"])

            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                optimizer_name=self.config.optimizer,
                learning_rate=self.config.lr_start_final,
                weight_decay=self.config.weight_decay,
                lr_final=self.config.lr_end_final,
                max_epochs=self.config.n_epochs_final,
            )

            start_epoch = 0
            if resume_state is not None:
                model.load_state_dict(resume_state["model_state_dict"])
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                if scheduler is not None and resume_state.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                start_epoch = int(resume_state.get("epoch", 0))

            for epoch in range(start_epoch, self.config.n_epochs_final):
                model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_total = 0

                if hasattr(model, "drop_path_prob"):
                    drop_path_max = float(getattr(model, "drop_path_prob"))
                    ratio = min(1.0, float(epoch + 1) / float(max(1, self.config.n_epochs_final)))
                    drop_prob = drop_path_max * ratio
                    if hasattr(model, "set_drop_path_prob") and callable(getattr(model, "set_drop_path_prob")):
                        model.set_drop_path_prob(drop_prob)
                    else:
                        setattr(model, "drop_path_prob", drop_prob)

                progress_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{self.config.n_epochs_final}",
                    leave=False,
                )
                for images, labels in progress_bar:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type="cuda" if self.device.type == "cuda" else "cpu",
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        outputs = model(images)
                        if (
                            self.config.auxiliary_loss_weight
                            and isinstance(outputs, (tuple, list))
                            and len(outputs) == 2
                        ):
                            logits, aux_logits = outputs
                            loss_main = criterion(logits, labels)
                            loss_aux = criterion(aux_logits, labels)
                            loss = loss_main + self.config.auxiliary_loss_weight * loss_aux
                        else:
                            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                            loss = criterion(logits, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    batch_size = labels.size(0)
                    train_loss_sum += loss.item() * batch_size
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_total += batch_size

                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{train_correct / max(1, train_total):.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    )

                    if self.config.developer_mode:
                        break

                train_loss = train_loss_sum / max(1, train_total)
                train_acc = train_correct / max(1, train_total)

                val_loss = None
                val_acc = None
                if valid_loader is not None:
                    model.eval()
                    val_loss_sum = 0.0
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for images, labels in valid_loader:
                            images = images.to(self.device, non_blocking=True)
                            labels = labels.to(self.device, non_blocking=True)
                            with torch.autocast(
                                device_type="cuda" if self.device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda",
                            ):
                                outputs = model(images)
                                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                                loss = criterion(logits, labels)

                            batch_size = labels.size(0)
                            val_loss_sum += loss.item() * batch_size
                            preds = logits.argmax(dim=1)
                            val_correct += (preds == labels).sum().item()
                            val_total += batch_size

                            if self.config.developer_mode:
                                break

                    val_loss = val_loss_sum / max(1, val_total)
                    val_acc = val_correct / max(1, val_total)

                metrics_callback.epochs.append(epoch)
                metrics_callback.train_losses.append(train_loss)
                metrics_callback.train_accs.append(train_acc)
                metrics_callback.lrs.append(optimizer.param_groups[0]["lr"])
                if val_loss is not None:
                    metrics_callback.val_losses.append(val_loss)
                if val_acc is not None:
                    metrics_callback.val_accs.append(val_acc)
                metrics_callback._has_data = True

                print(
                    f"[EPOCH {epoch + 1}/{self.config.n_epochs_final}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={(f'{val_loss:.4f}' if val_loss is not None else 'n/a')} "
                    f"val_acc={(f'{val_acc:.4f}' if val_acc is not None else 'n/a')} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

                if scheduler is not None:
                    scheduler.step()

                if (
                    checkpoint_path is not None
                    and self.config.checkpoint_every_n_epochs > 0
                    and (epoch + 1) % self.config.checkpoint_every_n_epochs == 0
                ):
                    save_training_checkpoint(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch + 1,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    print(
                        f"[CHECKPOINT] Saved periodic checkpoint at epoch {epoch + 1} to {checkpoint_path}"
                    )

                if self.config.developer_mode:
                    break

            model = model.to(self.device)
            time.sleep(1.0)

            if metrics_callback._has_data and len(metrics_callback.epochs) > 0:
                self._save_training_plot(metrics_callback, model_id)
            else:
                print(f"⏳ Model {model_id}: Waiting for metrics to be collected...")

            return model

        except Exception as e:
            print(
                f"[TRAIN EXCEPTION] model_id={model_id} pid={os.getpid()} "
                f"num_workers={getattr(train_loader, 'num_workers', 'n/a')} error={str(e)}"
            )
            print(f"Error training model {model_id}: {str(e)}")
            self._save_error_info(model_id, str(e))
            return None

    def _save_training_plot(self, metrics_callback, model_id: int):
        if not metrics_callback._has_data or not metrics_callback.epochs:
            print(f"⚠️ Model {model_id}: Insufficient data for plotting. Skipping.")
            return False

        n_epochs = len(metrics_callback.epochs)
        log_dir = "logs/train_plots"
        os.makedirs(log_dir, exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Training Metrics for Model {model_id} (Total Epochs: {n_epochs})",
                fontsize=16,
            )

            # 1. Losses
            axes[0, 0].plot(
                metrics_callback.epochs,
                metrics_callback.train_losses,
                "b-",
                label="Train Loss",
                linewidth=2,
            )
            if metrics_callback.val_losses:
                val_losses = metrics_callback.val_losses[:n_epochs] + [
                    metrics_callback.val_losses[-1]
                ] * max(0, n_epochs - len(metrics_callback.val_losses))
                axes[0, 0].plot(
                    metrics_callback.epochs,
                    val_losses,
                    "r-",
                    label="Validation Loss",
                    linewidth=2,
                )
            axes[0, 0].set_title("Training and Validation Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend(loc="best")
            axes[0, 0].grid(True, linestyle="--", alpha=0.7)

            # 2. Accuracies
            axes[0, 1].plot(
                metrics_callback.epochs,
                metrics_callback.train_accs,
                "g-",
                label="Train Accuracy",
                linewidth=2,
            )
            if metrics_callback.val_accs:
                val_accs = metrics_callback.val_accs[:n_epochs] + [
                    metrics_callback.val_accs[-1]
                ] * max(0, n_epochs - len(metrics_callback.val_accs))
                axes[0, 1].plot(
                    metrics_callback.epochs,
                    val_accs,
                    "m-",
                    label="Validation Accuracy",
                    linewidth=2,
                )
            axes[0, 1].set_title("Training and Validation Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_ylim(
                0,
                max(
                    1.0,
                    max(
                        metrics_callback.train_accs
                        + metrics_callback.val_accs[:n_epochs]
                        + [0.1]
                    )
                    * 1.1,
                ),
            )
            axes[0, 1].legend(loc="best")
            axes[0, 1].grid(True, linestyle="--", alpha=0.7)

            # 3. Learning Rate
            if metrics_callback.lrs:
                lrs = metrics_callback.lrs[:n_epochs] + [
                    metrics_callback.lrs[-1]
                ] * max(0, n_epochs - len(metrics_callback.lrs))
                axes[1, 0].plot(
                    metrics_callback.epochs,
                    lrs,
                    "c-",
                    label="Learning Rate",
                    linewidth=2,
                )
                axes[1, 0].set_title("Learning Rate Schedule")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("LR")
                if max(lrs) > min(lrs) * 10:
                    axes[1, 0].set_yscale("log")
                axes[1, 0].legend(loc="best")
                axes[1, 0].grid(True, linestyle="--", alpha=0.7)
            else:
                axes[1, 0].text(
                    0.5, 0.5, "No LR data available", ha="center", va="center"
                )
                axes[1, 0].set_title("Learning Rate")

            # 4. Summary Table
            metrics_summary = [
                ["Metric", "Final Value"],
                [
                    "Train Loss",
                    (
                        f"{metrics_callback.train_losses[-1]:.4f}"
                        if metrics_callback.train_losses
                        else "N/A"
                    ),
                ],
                [
                    "Val Loss",
                    (
                        f"{metrics_callback.val_losses[-1]:.4f}"
                        if metrics_callback.val_losses
                        else "N/A"
                    ),
                ],
                [
                    "Train Acc",
                    (
                        f"{metrics_callback.train_accs[-1]:.4f}"
                        if metrics_callback.train_accs
                        else "N/A"
                    ),
                ],
                [
                    "Val Acc",
                    (
                        f"{metrics_callback.val_accs[-1]:.4f}"
                        if metrics_callback.val_accs
                        else "N/A"
                    ),
                ],
                [
                    "Final LR",
                    (
                        f"{metrics_callback.lrs[-1]:.6f}"
                        if metrics_callback.lrs
                        else "N/A"
                    ),
                ],
            ]

            axes[1, 1].axis("tight")
            axes[1, 1].axis("off")
            table = axes[1, 1].table(
                cellText=metrics_summary[1:],
                colLabels=metrics_summary[0],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title("Final Metrics Summary", fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(
                log_dir, f"model_{model_id}_training_{timestamp}.png"
            )

            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return True

        except Exception as e:
            print(f"❌ Error generating plot: {str(e)}")
            raw_path = os.path.join(
                log_dir,
                f'model_{model_id}_raw_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz',
            )
            np.savez(
                raw_path,
                epochs=metrics_callback.epochs,
                train_losses=metrics_callback.train_losses,
                val_losses=metrics_callback.val_losses,
                train_accs=metrics_callback.train_accs,
                val_accs=metrics_callback.val_accs,
                lrs=metrics_callback.lrs,
            )
            print(f"🔧 Raw metrics saved for debugging: {raw_path}")
            return False

    def _save_error_info(self, model_id: int, error_msg: str):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        error_path = os.path.join(log_dir, f"error_model_{model_id}.txt")
        with open(error_path, "w") as f:
            f.write(f"Error occurred during training of model {model_id}:\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error details: {error_msg}\n")
            f.write(f"Full traceback:\n")
            traceback.print_exc(file=f)

        print(f"Error details saved to: {error_path}")

    def evaluate_and_save_results(
        self,
        model: torch.nn.Module,
        architecture: dict,
        data_loader,
        folder_name: str,
        model_id: int,
        mode: str = "classes",
    ) -> None:
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
        Finalizes the evaluation of the ensemble: calculates metrics and saves the results.

        Args:
            stats: Dictionary with statistics from collect_ensemble_stats
            file_name: The base file name for saving results

        Returns:
            Tuple: (ensemble_accuracy, model_accuracies, ece)
        """
        if stats is None:
            print("No stats provided for evaluation.")
            return None, None, None

        total = stats["total"]
        correct_ensemble = stats["correct_ensemble"]
        correct_models = stats["correct_models"]
        num_models = stats["num_models"]

        ensemble_acc = 100.0 * correct_ensemble / total if total > 0 else 0.0
        model_accs = [100.0 * c / total if total > 0 else 0.0 for c in correct_models]
        avg_model_acc = stats["avg_model_accuracy"] * 100.0

        ece = calculate_ece(stats)
        nll = calculate_nll(stats)
        oracle_nll = calculate_oracle_nll(stats)
        brier_score = calculate_brier_score(stats)

        predictive_disagreement = stats.get("predictive_disagreement", float("nan"))
        normalized_predictive_disagreement = stats.get(
            "normalized_predictive_disagreement", float("nan")
        )
        ambiguity = stats.get("ambiguity", float("nan"))

        fgsm_results = stats.get("fgsm_results", {})
        bim_results = stats.get("bim_results", {})
        pgd_results = stats.get("pgd_results", {})

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

            f.write("\n" + "=" * 60 + "\n")
            f.write("ADDITIONAL INFORMATION\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"Ensemble improvement over avg model: {ensemble_acc - avg_model_acc:.2f}%\n"
            )
            f.write(f"NLL improvement (Oracle - Ensemble): {oracle_nll - nll:.4f}\n")

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
        experiment_num = 0
        while (Path(path) / f"{file_name}_{experiment_num}.txt").exists():
            experiment_num += 1
        return experiment_num

    def _get_free_file_index_dir(self, path: str, dir_name: str) -> int:
        experiment_num = 1
        while (Path(path) / f"{dir_name}_{experiment_num}").exists():
            experiment_num += 1
        return experiment_num

    def get_latest_index_from_dir(self, base_path: Optional[Path] = None) -> int:
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
        checkpoint_dir,
        train_indices: List[int],
        valid_indices: List[int],
    ):
        """The target process for parallel learning."""
        local_config = copy.deepcopy(config)
        local_config.num_workers = 0
        if torch.cuda.is_available():
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_devices:
                visible_gpu_ids = [
                    int(x.strip()) for x in visible_devices.split(",") if x.strip()
                ]
                try:
                    local_gpu_id = visible_gpu_ids.index(physical_gpu_id)
                except ValueError as exc:
                    raise ValueError(
                        f"physical_gpu_id={physical_gpu_id} is not present in "
                        f"CUDA_VISIBLE_DEVICES='{visible_devices}'"
                    ) from exc
            else:
                local_gpu_id = physical_gpu_id
            local_config.device = f"cuda:{local_gpu_id}"
            torch.cuda.set_device(local_gpu_id)
            torch.cuda.empty_cache()
            print(
                f"[GPU MAP][train] physical_gpu_id={physical_gpu_id}, "
                f"CUDA_VISIBLE_DEVICES='{visible_devices}', local_gpu_id={local_gpu_id}"
            )
        else:
            local_config.device = "cpu"

        torch.set_float32_matmul_precision("high")
        device = torch.device(local_config.device)

        runner = DiversityNESRunner(
            local_config, DatasetsInfo.get(local_config.dataset_name.lower())
        )
        train_loader, valid_loader, test_loader = runner.get_data_loaders(
            train_indices=train_indices,
            valid_indices=valid_indices,
            download=False,
        )

        print(
            f"[PROCESS TRAIN] model_id={model_id} pid={os.getpid()} "
            f"forced_num_workers={local_config.num_workers} valid_loader_len={len(valid_loader)}"
        )

        final_model_path = Path(pth_path) / f"model_{model_id}.pth"
        checkpoint_path = Path(checkpoint_dir) / f"model_{model_id}.ckpt"

        if final_model_path.exists():
            print(
                f"[PROCESS TRAIN] model_id={model_id} final weights already exist at {final_model_path}, skipping training"
            )
            return

        model = runner.train_model(
            architecture,
            train_loader,
            None,
            model_id,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=checkpoint_path.exists(),
        )
        if model is None:
            print(f"[PROCESS TRAIN] model_id={model_id} training failed, skipping save/eval")
            return
        model = model.to(device)

        torch.save(model.state_dict(), final_model_path)
        print(f"[GPU {physical_gpu_id}] Model {model_id} saved to {pth_path}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[CHECKPOINT] Removed resume checkpoint {checkpoint_path}")

        eval_loader = test_loader if config.evaluate_ensemble_flag else valid_loader
        runner.evaluate_and_save_results(
            model, architecture, eval_loader, str(archs_path), model_id=model_id
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fill_models_list(self, archs_path, pth_path) -> None:
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
                print(f"Model {model_id} loaded")

            except Exception as e:
                print(f"Failed to load {json_file}: {e}")

        print(f"Loaded {len(self.models)} models.")

    def run_pretrained(self, index: int) -> None:
        """Loads and evaluates pretrained models in parallel by index."""
        root_dir = Path(self.config.best_models_save_path)
        json_dir = root_dir / f"trained_models_archs_{index}"
        pth_dir = root_dir / f"trained_models_pth_{index}"

        train_data, _ = self.prepare_datasets(download=True)
        train_indices, valid_indices = self.build_split_indices(len(train_data))

        if not json_dir.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")
        if not pth_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pth_dir}")

        print(f"Using JSON from: {json_dir}, PTH from: {pth_dir}")
        arch_dicts = load_json_from_directory(json_dir)
        if not arch_dicts:
            raise RuntimeError("No architectures found")

        eval_output_dir = (
            Path(self.config.output_path) / f"trained_models_archs_{index}"
        )
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        model_tasks = []
        for entry in arch_dicts:
            arch = entry["architecture"]
            model_id = entry.get("id")
            if model_id is None:
                raise ValueError("Missing 'id' in architecture")
            pth_file = pth_dir / f"model_{model_id}.pth"
            if not pth_file.exists():
                print(f"Skipping model {model_id}: weights not found at {pth_file}")
                continue
            model_tasks.append((arch, model_id, pth_file))

        if not model_tasks:
            print("No valid models to evaluate.")
            return

        n_models = len(model_tasks)
        available_gpus = self._get_available_gpus()
        n_gpus = len(available_gpus)
        max_per_gpu = self.config.max_per_gpu

        print(f"Available GPUs: {available_gpus}")
        print(f"Evaluating {n_models} models, up to {max_per_gpu} per GPU")

        processes = []
        active_processes = []

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
                    train_indices,
                    valid_indices,
                ),
            )
            p.start()
            active_processes.append((p, model_id))
            print(f"Started evaluation of model {model_id} on GPU {physical_gpu_id}")

            while len(active_processes) >= n_gpus * max_per_gpu:
                for proc, mid in active_processes[:]:
                    if not proc.is_alive():
                        proc.join()
                        active_processes.remove((proc, mid))
                        break
                time.sleep(0.5)

        completed = 0
        with tqdm(total=n_models, desc="Evaluating models") as pbar:
            while active_processes:
                for proc, mid in active_processes[:]:
                    if not proc.is_alive():
                        proc.join()
                        active_processes.remove((proc, mid))
                        completed += 1
                        pbar.update(1)
                time.sleep(0.5)

        print(f"✅ All models from index {index} evaluated!")

        if self.config.evaluate_ensemble_flag:
            print("Evaluating ensemble for pretrained models...")
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
                print("No models loaded for ensemble evaluation.")

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
        train_indices: List[int],
        valid_indices: List[int],
    ):
        """The process of evaluating a single pre-trained model."""
        local_config = copy.deepcopy(config)
        local_config.num_workers = 0

        try:
            if torch.cuda.is_available():
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if visible_devices:
                    visible_gpu_ids = [
                        int(x.strip()) for x in visible_devices.split(",") if x.strip()
                    ]
                    try:
                        local_gpu_id = visible_gpu_ids.index(physical_gpu_id)
                    except ValueError as exc:
                        raise ValueError(
                            f"physical_gpu_id={physical_gpu_id} is not present in "
                            f"CUDA_VISIBLE_DEVICES='{visible_devices}'"
                        ) from exc
                else:
                    local_gpu_id = physical_gpu_id
                local_config.device = f"cuda:{local_gpu_id}"
                torch.cuda.set_device(local_gpu_id)
                torch.cuda.empty_cache()
                print(
                    f"[GPU MAP][eval] physical_gpu_id={physical_gpu_id}, "
                    f"CUDA_VISIBLE_DEVICES='{visible_devices}', local_gpu_id={local_gpu_id}"
                )
            else:
                local_config.device = "cpu"
            torch.set_float32_matmul_precision("high")

            device = torch.device(local_config.device)

            info = DatasetsInfo.get(local_config.dataset_name.lower())
            runner = DiversityNESRunner(local_config, info)
            _, valid_loader, test_loader = runner.get_data_loaders(
                train_indices=train_indices,
                valid_indices=valid_indices,
                download=False,
            )

            if config.evaluate_ensemble_flag:
                eval_loader = test_loader
            else:
                eval_loader = valid_loader

            with model_context(architecture):
                model = DartsSpace(
                    width=config.width,
                    num_cells=config.num_cells,
                    dataset=dataset_key,
                )
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)

            runner.evaluate_and_save_results(
                model,
                architecture,
                eval_loader,
                folder_name=eval_output_dir,
                model_id=model_id,
                mode="class",
            )

        except Exception as e:
            print(f"Error evaluating model {model_id} on GPU {physical_gpu_id}: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_models_from_index(self, index: int) -> None:
        """Loads all models from the specified index for the ensemble."""
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
        """Evaluates all saved ensembles."""
        root_dir = Path(self.config.best_models_save_path)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        last_index = self._get_free_file_index_dir(root_dir, "trained_models_archs")
        print(f"Last index: {last_index}")
        for idx in range(1, last_index + 1):
            if (root_dir / f"trained_models_archs_{idx}").exists() and (
                root_dir / f"trained_models_pth_{idx}"
            ).exists():
                self.run_pretrained(idx)
                self.models = []
            else:
                print(f"Skipping index {idx}: missing directories.")

    def run(self) -> None:

        train_data, _ = self.prepare_datasets(download=True)
        train_indices, valid_indices = self.build_split_indices(len(train_data))

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
            checkpoint_dir = output_path / f"checkpoints_{cur_index}"
            archs_and_pth_path_list.append((archs_path, pth_path))
            archs_path.mkdir(parents=True, exist_ok=True)
            pth_path.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        processes = []

        for idx, (cur_index, arch) in enumerate(archs_list):
            gpu_idx = (idx // max_per_gpu) % n_gpus
            physical_gpu_id = available_gpus[gpu_idx]
            model_id = idx % index_2_n_archs[cur_index]

            archs_path = output_path / f"trained_models_archs_{cur_index}"
            pth_path = output_path / f"trained_models_pth_{cur_index}"
            checkpoint_dir = output_path / f"checkpoints_{cur_index}"

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
                    checkpoint_dir,
                    train_indices,
                    valid_indices,
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
        print("All models trained!")

        if self.config.evaluate_ensemble_flag:
            print("Evaluating ensemble...")
            for cur_index, (archs_path, pth_path) in enumerate(archs_and_pth_path_list):
                self.models = []
                self.fill_models_list(archs_path, pth_path)
                if self.models:
                    _, _, test_loader = self.get_data_loaders(
                        train_indices=train_indices,
                        valid_indices=valid_indices,
                        download=False,
                    )
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
                    print("No models to evaluate.")
        else:
            print("Ensemble evaluation skipped.")

        print("Training and evaluation completed!")

    def _get_available_gpus(self) -> List[int]:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        return list(range(torch.cuda.device_count()))


# === ENTRY POINT ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTS NAS Runner")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        required=True,
        help="Path to JSON file with TrainConfig parameters",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=None,
        help="Save per-model resume checkpoints every N epochs. 0 disables periodic checkpoints.",
    )
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    if args.checkpoint_every_n_epochs is not None:
        params["checkpoint_every_n_epochs"] = args.checkpoint_every_n_epochs
    params["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert params["seed"] != -1, "Seed must be set!"

    config = TrainConfig(**params)
    info = DatasetsInfo.get(config.dataset_name.lower())
    runner = DiversityNESRunner(config, info)

    if config.use_pretrained_models_for_ensemble:
        runner.run_all_pretrained()
    else:
        runner.run()
