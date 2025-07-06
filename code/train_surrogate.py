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
        for file_path in tqdm(
            self.dataset_path.rglob("*.json"), desc="Loading dataset"
        ):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                self.config.models_dict.append(data)
            except json.JSONDecodeError as e:
                logger.error("Error decoding JSON from %s: %s", file_path, e)
        if len(self.config.models_dict) < self.config.n_models:
            raise ValueError(
                f"Only {len(self.config.models_dict)} models found, but n_models={self.config.n_models}"
            )

        self.config.models_dict = random.sample(
            self.config.models_dict, self.config.n_models
        )

    def _prepare_predictions(self, num_samples: Optional[int] = None):
        preds = []
        for data in self.config.models_dict:
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

    def convert_into_graphs(self) -> None:
        self.config.graphs = [
            Graph(d, index=i) for i, d in enumerate(self.config.models_dict)
        ]

    def create_datasets(self) -> None:
        accs = [d["test_accuracy"] for d in self.config.models_dict]
        ds = CustomDataset(self.config.graphs, accs)
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
            num_workers=4,
            collate_fn=collate_triplets,
        )
        cfg.valid_loader_diversity = DataLoader(
            cfg.valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_triplets,
        )
        cfg.train_loader_accuracy = DataLoader(
            cfg.base_train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_graphs,
        )
        cfg.valid_loader_accuracy = DataLoader(
            cfg.base_valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
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
    trainer.load_dataset()
    trainer.get_diversity_matrix()
    trainer.create_discrete_diversity_matrix()
    trainer.convert_into_graphs()
    trainer.create_datasets()
    trainer.create_dataloaders()
    trainer.train_accuracy_model()
    trainer.train_diversity_model()
    trainer.save_models()
