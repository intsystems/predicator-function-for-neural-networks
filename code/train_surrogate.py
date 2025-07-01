import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Custom imports
import sys

sys.path.insert(1, "../dependencies")

from dependencies.Graph import Graph
from dependencies.GCN import (
    GAT,
    CustomDataset,
    TripletGraphDataset,
    train_model_accuracy,
    train_model_diversity,
    collate_triplets,
    collate_graphs,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DARTS_OPS = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

@dataclass
class TrainConfig:
    # Paths and device
    dataset_path: str
    device: str  # e.g., 'cuda' or 'cpu'

    # General settings
    developer_mode: bool
    n_models: int

    # Diversity margins
    upper_margin: float
    lower_margin: float

    # Data and batching
    train_size: float  # fraction in [0, 1]
    batch_size: int

    # Accuracy model hyperparameters
    acc_num_epochs: int
    acc_lr: float
    acc_final_lr: float
    acc_dropout: float
    acc_n_heads: int
    draw_fig_acc: bool

    # Diversity model hyperparameters
    div_num_epochs: int
    div_lr: float
    div_final_lr: float
    div_dropout: float
    div_n_heads: int
    margin: float
    output_dim: int
    draw_fig_div: bool

    # Optional fields initialized later
    model_accuracy: Optional[Any] = field(default=None, init=False)
    model_diversity: Optional[Any] = field(default=None, init=False)
    models_dict: Dict[int, Dict] = field(default_factory=dict, init=False)
    graphs: Optional[Any] = field(default=None, init=False)
    diversity_matrix: Optional[Any] = field(default=None, init=False)
    discrete_diversity_matrix: Optional[Any] = field(default=None, init=False)
    base_train_dataset: Optional[Any] = field(default=None, init=False)
    base_valid_dataset: Optional[Any] = field(default=None, init=False)
    train_dataset: Optional[Any] = field(default=None, init=False)
    valid_dataset: Optional[Any] = field(default=None, init=False)
    full_triplet_dataset: Optional[Any] = field(default=None, init=False)
    train_loader_diversity: Optional[Any] = field(default=None, init=False)
    valid_loader_diversity: Optional[Any] = field(default=None, init=False)
    train_loader_accuracy: Optional[Any] = field(default=None, init=False)
    valid_loader_accuracy: Optional[Any] = field(default=None, init=False)


class SurrogateTrainer:
    def __init__(self, config: TrainConfig):
        """
        Trainer for surrogate models.

        Args:
            config: TrainConfig instance with all hyperparameters.
        """
        self.config = config
        self.device = torch.device(self.config.device)
        self.dataset_path = self.config.dataset_path

    def load_dataset(self):
        self.models_dict = []
        for root, _, files in os.walk(self.dataset_path):
            for file in tqdm(files, desc="Loading dataset"):
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            self.models_dict.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file {file_path}: {e}")

    def _prepare_predictions(self, num_samples=None):
        cached_preds = {}

        for i in range(self.config.n_models):
            preds = np.array(self.models_dict[i]["test_predictions"])
            if num_samples is not None:
                preds = preds[:num_samples]
            cached_preds[i] = preds

        return cached_preds

    def get_diversity_matrix_naive(self, num_samples=None):
        self.diversity_matrix = np.eye(self.config.n_models)

        cached_preds = self._prepare_predictions(num_samples)

        for i in tqdm(range(self.config.n_models), desc="Computing diversity matrix"):
            for j in range(i + 1, self.config.n_models):
                distance = np.mean(cached_preds[i] == cached_preds[j])
                self.diversity_matrix[i, j] = self.diversity_matrix[j, i] = distance

    def get_diversity_matrix(self, num_samples=None):
        self.diversity_matrix = np.eye(self.config.n_models)

        cached_preds = self._prepare_predictions(num_samples)

        for i in tqdm(range(self.config.n_models), desc="Computing diversity matrix"):
            for j in range(i + 1, self.config.n_models):
                distance = np.mean(
                    np.sqrt(jensenshannon(cached_preds[i], cached_preds[j]))
                )
                self.diversity_matrix[i, j] = self.diversity_matrix[j, i] = distance

    def create_discrete_diversity_matrix(self):
        self.discrete_diversity_matrix = np.zeros((self.config.n_models, self.config.n_models))

        upper_margins = np.quantile(self.diversity_matrix, self.config.upper_margin, axis=1)
        lower_margins = np.quantile(self.diversity_matrix, self.config.lower_margin, axis=1)

        self.discrete_diversity_matrix[
            self.diversity_matrix > upper_margins[:, None]
        ] = 1
        self.discrete_diversity_matrix[
            self.diversity_matrix < lower_margins[:, None]
        ] = -1

    def convert_into_graphs(self):
        self.graphs = [
            Graph(model_dict, index=i)
            for (i, model_dict) in enumerate(self.models_dict)
        ]

    def create_datasets(self):
        accuracies = [model["test_accuracy"] for model in self.models_dict]
        graphs_dataset = CustomDataset(self.graphs, accuracies)

        train_size = int(self.config.train_size * self.config.n_models)
        valid_size = self.config.n_models - train_size

        self.base_train_dataset, self.base_valid_dataset = random_split(
            graphs_dataset, [train_size, valid_size]
        )

        self.train_dataset = TripletGraphDataset(
            self.base_train_dataset, self.discrete_diversity_matrix
        )
        self.valid_dataset = TripletGraphDataset(
            self.base_valid_dataset, self.discrete_diversity_matrix
        )
        self.full_triplet_dataset = TripletGraphDataset(
            graphs_dataset, self.discrete_diversity_matrix
        )

    def create_dataloaders(self):
        self.train_loader_diversity = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_triplets,
        )

        self.valid_loader_diversity = DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_triplets,
        )

        self.train_loader_accuracy = DataLoader(
            self.base_train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_graphs,
        )

        self.valid_loader_accuracy = DataLoader(
            self.base_valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_graphs,
        )

    def train_accuracy_model(self):
        input_dim = self.base_train_dataset[0]["x"].shape[1]
        output_dim = 1

        self.model_accuracy = GAT(
            input_dim,
            output_dim=output_dim,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
        )
        optimizer = torch.optim.AdamW(self.model_accuracy.parameters(), lr=self.config.acc_lr)
        criterion = nn.MSELoss()

        train_model_accuracy(
            self.model_accuracy,
            self.train_loader_accuracy,
            self.valid_loader_accuracy,
            optimizer,
            criterion,
            self.config.acc_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.acc_final_lr,
            draw_figure=self.config.draw_fig_acc,
        )

    def _triplet_loss(self, anchor, positive, negative, margin=0.1):
        d_ap = (anchor - positive).pow(2).sum(-1)
        d_an = (anchor - negative).pow(2).sum(-1)

        loss = F.relu(d_ap - d_an + margin)
        return loss.mean()

    def train_diversity_model(self):
        input_dim = self.base_train_dataset[0]["x"].shape[1]
        output_dim = self.config.output_dim

        self.model_diversity = GAT(
            input_dim, output_dim, dropout=self.config.div_dropout, heads=self.config.div_n_heads
        )
        optimizer = torch.optim.AdamW(self.model_diversity.parameters(), lr=self.config.div_lr)
        criterion = lambda anchor, positive, negative: self._triplet_loss(
            anchor, positive, negative, margin=self.config.margin
        )

        train_model_diversity(
            self.model_diversity,
            self.train_loader_diversity,
            self.valid_loader_diversity,
            optimizer,
            criterion,
            self.config.div_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.div_final_lr,
            draw_figure=self.config.draw_fig_div,
        )

    def save_models(self, dir_name="surrogate_models"):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(
            self.model_accuracy.state_dict(),
            os.path.join(dir_name, "model_accuracy.pth"),
        )
        torch.save(
            self.model_diversity.state_dict(),
            os.path.join(dir_name, "model_diversity.pth"),
        )


if __name__ == "__main__":
    # Парсинг аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        required=True,
        help="Path to the hyperparameters JSON",
    )
    args = parser.parse_args()

    # Выбираем устройство и приводим к строке для TrainConfig
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем JSON-гиперпараметры
    with open(args.hyperparameters_json, "r") as f:
        hp: dict = json.load(f)

    # Строим конфиг dataclass, передавая dataset_path и device отдельно,
    # а остальные поля распаковываем из hp
    config = TrainConfig(
        **hp
    )

    # Создаём тренера и запускаем пайплайн
    surrogate_trainer = SurrogateTrainer(config)
    surrogate_trainer.load_dataset()
    surrogate_trainer.get_diversity_matrix_naive()
    surrogate_trainer.create_discrete_diversity_matrix()
    surrogate_trainer.convert_into_graphs()
    surrogate_trainer.create_datasets()
    surrogate_trainer.create_dataloaders()
    surrogate_trainer.train_accuracy_model()
    surrogate_trainer.train_diversity_model()
    surrogate_trainer.save_models()

