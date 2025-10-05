import json
import logging
import random
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit, prange
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from scipy.spatial import distance
import shutil
import matplotlib.pyplot as plt

# Custom imports
import sys

sys.path.insert(1, "../dependencies")

from dependencies.data_generator import load_dataset
from dependencies.train_config import TrainConfig
from dependencies.GCN import (
    GAT_ver_1,
    GAT_ver_2,
    CustomDataset,
    TripletGraphDataset,
    train_model_accuracy,
    train_model_diversity,
    collate_triplets,
    collate_graphs,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurrogateTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dataset_path = Path(config.dataset_path)
        self.acc_n_models = len(config.acc_models_dict_path)
        self.div_n_models = len(config.div_models_dict_path)

    def _prepare_predictions(self, num_samples: Optional[int] = None):
        preds = []
        for path in tqdm(
            self.config.div_models_dict_path, desc="Preparing predictions"
        ):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            prediction_name = (
                "valid_predictions"
                if "valid_predictions" in data
                else "test_predictions"
            )
            arr = np.array(data[prediction_name])
            preds.append(arr[:num_samples] if num_samples else arr)
        return preds

    def get_diversity_matrix(self, num_samples: Optional[int] = 5500) -> None:
        n = self.div_n_models
        preds = self._prepare_predictions(num_samples)

        if self.config.diversity_matrix_metric == "overlap":
            arr = np.stack(preds)  # shape: (n, num_samples)
            self.config.diversity_matrix = compute_overlap_diversity_matrix(arr)
        elif self.config.diversity_matrix_metric == "js":
            arr = np.stack(preds)  # shape: (n, num_samples, n_classes)
            self.config.diversity_matrix = compute_js_diversity_matrix(arr)
        else:
            self.config.diversity_matrix = np.eye(n)

        self.log_diversities()

    def log_diversities(self):
        os.makedirs("logs/", exist_ok=True)
        diversities = []
        for i in tqdm(range(self.div_n_models)):
            for j in range(i + 1, self.div_n_models):
                diversities.append(self.config.diversity_matrix[i, j])
        diversities = np.array(diversities)

        plt.figure(figsize=(10, 6))

        plt.hist(
            diversities,
            bins=50,
            edgecolor="black",
            weights=np.ones(len(diversities)) / len(diversities),
        )
        # plt.title("Distribution of Model Diversity")
        plt.xlabel("Percentage of Differences", fontsize=18)
        plt.ylabel("Percentage", fontsize=18)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig("logs/diversities.png")
        plt.close()

    def create_discrete_diversity_matrix(self) -> None:
        M = self.config.diversity_matrix
        upper = np.quantile(M, self.config.upper_margin, axis=1)
        lower = np.quantile(M, self.config.lower_margin, axis=1)
        D = np.zeros_like(M)
        D[M > upper[:, None]] = 1
        D[M < lower[:, None]] = -1
        self.config.discrete_diversity_matrix = D

    def get_accuracies(self, acc=True):
        accs = []
        dict_path = (
            self.config.acc_models_dict_path
            if acc
            else self.config.div_models_dict_path
        )
        for path in dict_path:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            accs.append(
                data["valid_accuracy"]
                if "valid_accuracy" in data
                else data["test_accuracy"]
            )
        if acc:
            self.log_accuracies(accs)
        accs = np.array(accs) * 100
        return accs

    def log_accuracies(self, accs):
        os.makedirs("logs/", exist_ok=True)
        plt.figure(figsize=(10, 6))

        plt.hist(
            accs,
            bins=50,
            edgecolor="black",
            color="green",
            weights=[1 / len(accs)] * len(accs),
        )
        # plt.title('Distribution of Model Accuracies')
        plt.xlabel("Accuracy", fontsize=18)
        plt.ylabel("Percentage", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig("logs/accuracies.png")
        plt.close()

    def create_datasets(self) -> None:
        # acc surrogate
        acc_accs = self.get_accuracies(acc=True)
        acc_ds = CustomDataset(self.config.acc_models_dict_path, acc_accs)
        acc_train_n = int(self.config.train_size * self.acc_n_models)
        self.config.base_train_dataset, self.config.base_valid_dataset = random_split(
            acc_ds, [acc_train_n, self.acc_n_models - acc_train_n]
        )

        # diversity surrogate triplets
        div_accs = self.get_accuracies(acc=False)
        div_ds = CustomDataset(self.config.div_models_dict_path, div_accs)
        div_train_n = int(self.config.train_size * self.div_n_models)
        div_train, div_valid = random_split(
            div_ds, [div_train_n, self.div_n_models - div_train_n]
        )
        div_train_triplet = TripletGraphDataset(
            div_train, self.config.discrete_diversity_matrix
        )
        div_valid_triplet = TripletGraphDataset(
            div_valid, self.config.discrete_diversity_matrix
        )
        self.config.train_dataset = div_train_triplet
        self.config.valid_dataset = div_valid_triplet
        self.config.full_triplet_dataset = TripletGraphDataset(
            div_ds, self.config.discrete_diversity_matrix
        )

    def create_dataloaders(self) -> None:
        cfg = self.config
        cfg.train_loader_diversity = DataLoader(
            cfg.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_triplets,
        )
        cfg.valid_loader_diversity = DataLoader(
            cfg.valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_triplets,
        )
        cfg.train_loader_accuracy = DataLoader(
            cfg.base_train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
        )
        cfg.valid_loader_accuracy = DataLoader(
            cfg.base_valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
        )

    def train_accuracy_model(self) -> None:
        self.config.model_accuracy = GAT_ver_2(
            self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
            output_activation="none",
            pre_norm=True,
            pooling="attn",
        )
        opt = torch.optim.AdamW(
            self.config.model_accuracy.parameters(), lr=self.config.acc_lr_start
        )
        train_model_accuracy(
            self.config.model_accuracy,
            self.config.train_loader_accuracy,
            self.config.valid_loader_accuracy,
            opt,
            nn.MSELoss(),
            self.config.acc_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.acc_lr_end,
        )

    def _triplet_loss(
        self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        d_ap = (a - p).pow(2).sum(-1)
        d_an = (a - n).pow(2).sum(-1)
        return F.relu(d_ap - d_an + self.config.margin).mean()

    def train_diversity_model(self) -> None:
        self.config.model_diversity = GAT_ver_2(
            self.config.input_dim,
            self.config.div_output_dim,
            dropout=self.config.div_dropout,
            heads=self.config.div_n_heads,
            output_activation="l2",
            pre_norm=True,
        )

        opt = torch.optim.AdamW(
            self.config.model_diversity.parameters(), lr=self.config.div_lr_start
        )
        train_model_diversity(
            self.config.model_diversity,
            self.config.train_loader_diversity,
            self.config.valid_loader_diversity,
            opt,
            lambda a, p, n: self._triplet_loss(a, p, n),
            self.config.div_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.div_lr_end,
        )

    def save_models(self) -> None:
        path = Path(self.config.surrogate_inference_path)
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.config.model_accuracy.state_dict(),
            path / "model_accuracy.pth",
        )
        torch.save(
            self.config.model_diversity.state_dict(),
            path / "model_diversity.pth",
        )


@njit(parallel=True)
def compute_overlap_diversity_matrix(preds):
    n = preds.shape[0]
    M = np.eye(n)
    for i in prange(n):
        for j in range(i + 1, n):
            dist = np.mean(preds[i] == preds[j])
            M[i, j] = M[j, i] = dist
    return M


@njit
def kl_divergence(p, q):
    kl = 0.0
    for i in range(len(p)):
        if p[i] > 0 and q[i] > 0:
            kl += p[i] * np.log(p[i] / q[i])
    return kl


@njit
def jensen_shannon_divergence(p, q):
    m = (p + q) * 0.5
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


@njit(parallel=True)
def compute_overlap_diversity_matrix(preds):
    n = preds.shape[0]
    M = np.eye(n)
    for i in prange(n):
        for j in range(i + 1, n):
            dist = np.mean(preds[i] == preds[j])
            M[i, j] = M[j, i] = dist
    return M


@njit
def compute_js_row(probs_i, probs_rest):
    """Вычисляет JS-дивергенции между probs_i и всеми в probs_rest."""
    n_models_rest, n_samples, n_classes = probs_rest.shape
    js_vals = np.empty(n_models_rest, dtype=np.float32)
    for j in range(n_models_rest):
        js_sum = 0.0
        for s in range(n_samples):
            js_sum += jensen_shannon_divergence(probs_i[s], probs_rest[j, s])
        js_vals[j] = js_sum / n_samples
    return js_vals


def compute_js_diversity_matrix(probs):
    n_models = probs.shape[0]
    M = np.zeros((n_models, n_models), dtype=np.float32)

    for i in tqdm(range(n_models), desc="Computing JS diversity matrix"):
        if i + 1 < n_models:
            row_vals = compute_js_row(probs[i], probs[i + 1 :])
            M[i, i + 1 :] = row_vals
            M[i + 1 :, i] = row_vals
    return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate trainer")
    parser.add_argument("--hyperparameters_json", type=str, required=True)
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    config = TrainConfig(**params)

    load_dataset(config)

    trainer = SurrogateTrainer(config)
    print("Loading models")
    print("Getting diversity matrix")
    trainer.get_diversity_matrix()
    print("Creating discrete diversity matrix")
    trainer.create_discrete_diversity_matrix()
    print("Creating datasets")
    trainer.create_datasets()
    print("Creating dataloaders")
    trainer.create_dataloaders()
    print("Training diversity surrogate")
    trainer.train_diversity_model()
    print("Training accuracy surrogate")
    trainer.train_accuracy_model()
    print("Saving models")
    trainer.save_models()
