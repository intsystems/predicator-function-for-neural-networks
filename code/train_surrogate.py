import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from scipy.spatial import distance
import shutil

# Custom imports
import sys

sys.path.insert(1, "../dependencies")

from dependencies.Graph import Graph
from dependencies.train_config import TrainConfig
from dependencies.GCN import (
    GAT,
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

    def load_dataset(self) -> None:
        self.config.models_dict_path = []
        for i, file_path in enumerate(tqdm(
            self.dataset_path.rglob("*.json"), desc="Loading dataset"
        )):
            self.config.models_dict_path.append(file_path)

        if len(self.config.models_dict_path) < self.config.n_models:
            raise ValueError(
                f"Only {len(self.config.models_dict_path)} model paths found, but n_models={self.config.n_models}"
            )

        self.config.models_dict_path = random.sample(
            self.config.models_dict_path, self.config.n_models
        )

    def _prepare_predictions(self, num_samples: Optional[int] = None):
        preds = []
        for path in self.config.models_dict_path:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            arr = np.array(data["test_predictions"])
            preds.append(arr[:num_samples] if num_samples else arr)
        return preds

    def get_diversity_matrix(self, num_samples: Optional[int] = None) -> None:
        n = self.config.n_models
        self.config.diversity_matrix = np.eye(n)
        preds = self._prepare_predictions(num_samples)
        for i in tqdm(range(n), desc="Computing diversity matrix (i loop)"):
            for j in range(i + 1, n):
                if self.config.diversity_matrix_metric == "overlap":
                    dist = np.mean(preds[i] == preds[j])
                elif self.config.diversity_matrix_metric == "js":
                    dist = np.mean(np.sqrt(distance.jensenshannon(preds[i], preds[j])))
                self.config.diversity_matrix[i, j] = self.config.diversity_matrix[
                    j, i
                ] = dist

    def create_discrete_diversity_matrix(self) -> None:
        M = self.config.diversity_matrix
        upper = np.quantile(M, self.config.upper_margin, axis=1)
        lower = np.quantile(M, self.config.lower_margin, axis=1)
        D = np.zeros_like(M)
        D[M > upper[:, None]] = 1
        D[M < lower[:, None]] = -1
        self.config.discrete_diversity_matrix = D

    def get_accuracies(self):
        accs = []
        for path in self.config.models_dict_path:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            accs.append(data["test_accuracy"])
        return accs


    def create_datasets(self) -> None:
        accs = self.get_accuracies()
        ds = CustomDataset(self.config.models_dict_path, accs)
        train_n = int(self.config.train_size * self.config.n_models)
        self.config.base_train_dataset, self.config.base_valid_dataset = random_split(
            ds, [train_n, self.config.n_models - train_n]
        )
        self.config.train_dataset = TripletGraphDataset(
            self.config.base_train_dataset, self.config.discrete_diversity_matrix
        )
        self.config.valid_dataset = TripletGraphDataset(
            self.config.base_valid_dataset, self.config.discrete_diversity_matrix
        )
        self.config.full_triplet_dataset = TripletGraphDataset(
            ds, self.config.discrete_diversity_matrix
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
        self.config.model_accuracy = GAT(
            self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
        )
        opt = torch.optim.AdamW(
            self.config.model_accuracy.parameters(), lr=self.config.acc_lr
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
            final_lr=self.config.acc_final_lr,
            draw_figure=self.config.draw_fig_acc,
        )

    def _triplet_loss(
        self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        d_ap = (a - p).pow(2).sum(-1)
        d_an = (a - n).pow(2).sum(-1)
        return F.relu(d_ap - d_an + self.config.margin).mean()

    def train_diversity_model(self) -> None:
        self.config.model_diversity = GAT(
            self.config.input_dim,
            self.config.div_output_dim,
            dropout=self.config.div_dropout,
            heads=self.config.div_n_heads,
        )
        opt = torch.optim.AdamW(
            self.config.model_diversity.parameters(), lr=self.config.div_lr
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
            final_lr=self.config.div_final_lr,
            draw_figure=self.config.draw_fig_div,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate trainer")
    parser.add_argument("--hyperparameters_json", type=str, required=True)
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    params.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    config = TrainConfig(**params)

    trainer = SurrogateTrainer(config)
    print("Loading models")
    trainer.load_dataset()
    print("Getting diversuty matrix")
    trainer.get_diversity_matrix()
    print("Creating discrete diversity matrix")
    trainer.create_discrete_diversity_matrix()
    print("Creating datasets")
    trainer.create_datasets()
    print("Creating dataloaders")
    trainer.create_dataloaders()
    print("Training accuracy surrogate")
    trainer.train_accuracy_model()
    print("Training diversity surrogate")
    trainer.train_diversity_model()
    print("Saving models")
    trainer.save_models()
