import sys
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
    GraphNorm,
)
from torch_geometric.utils import dense_to_sparse
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Graph import Graph

class SimpleGCN(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, hidden_dim=64, dropout=0.5, pooling="max"
    ):
        """
        input_dim: размер входных признаков узлов
        hidden_dim: размер скрытого пространства в графовых свёрточных слоях
        embedding_dim: размер итогового графового эмбеддинга
        dropout: вероятность исключения узлов для регуляризации
        pooling: тип агрегации ('max', 'mean' или 'sum')
        """
        super(SimpleGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)

        self.graph_norm = GraphNorm(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.gc1(x, edge_index))
        x = self.graph_norm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = F.relu(self.gc2(x, edge_index))
        x = self.graph_norm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Пулинг по графу
        if self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        x = self.fc(x)

        if self.embedding_dim == 1:
            x = torch.sigmoid(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim=16, dropout=0.5, pooling="max"):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64  # базовая размерность скрытого слоя

        self.gc1 = GCNConv(input_dim, self.hidden_dim)
        self.gc2 = GCNConv(self.hidden_dim, 256)
        self.gc3 = GCNConv(256, 512)
        self.gc4 = GCNConv(512, self.hidden_dim)

        self.residual_proj = (
            nn.Linear(input_dim, self.hidden_dim)
            if input_dim != self.hidden_dim
            else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_norm = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        residual = self.residual_proj(x)

        x = F.leaky_relu(self.gc1(x, edge_index))
        x = self.dropout(x)

        x = F.leaky_relu(self.gc2(x, edge_index))
        x = self.dropout(x)

        x = F.leaky_relu(self.gc3(x, edge_index))
        x = self.dropout(x)

        x = F.leaky_relu(self.gc4(x, edge_index))
        x = self.dropout(x)

        x = self.layer_norm(x + residual)

        # Используем global pooling в зависимости от выбранного метода
        if self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        if self.output_dim == 1:
            x = torch.sigmoid(x)
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim=16, dropout=0.5, pooling="max", heads=4):
        super(GAT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.heads = heads

        self.gat1 = GATv2Conv(input_dim, self.hidden_dim // heads, heads=heads)
        self.gat2 = GATv2Conv(self.hidden_dim, 256 // heads, heads=heads)
        self.gat3 = GATv2Conv(256, 256 // heads, heads=heads)
        self.gat4 = GATv2Conv(256, self.hidden_dim // heads, heads=heads)

        # residual projection
        self.res1 = nn.Linear(input_dim, self.hidden_dim)
        self.res2 = nn.Linear(self.hidden_dim, 256)
        self.res3 = nn.Linear(256, 256)
        self.res4 = nn.Linear(256, self.hidden_dim)

        self.norm1 = GraphNorm(self.hidden_dim)
        self.norm2 = GraphNorm(256)
        self.norm3 = GraphNorm(256)
        self.norm4 = GraphNorm(self.hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_norm = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # Layer 1
        h1 = self.gat1(x, edge_index)
        h1 = F.leaky_relu(h1 + self.res1(x))
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)

        # Layer 2
        h2 = self.gat2(h1, edge_index)
        h2 = F.leaky_relu(h2 + self.res2(h1))
        h2 = self.norm2(h2)
        h2 = self.dropout(h2)

        # Layer 3
        h3 = self.gat3(h2, edge_index)
        h3 = F.leaky_relu(h3 + self.res3(h2))
        h3 = self.norm3(h3)
        h3 = self.dropout(h3)

        # Layer 4
        h4 = self.gat4(h3, edge_index)
        h4 = F.leaky_relu(h4 + self.res4(h3))
        h4 = self.norm4(h4)
        h4 = self.dropout(h4)

        # Global pooling
        if self.pooling == "max":
            hg = global_max_pool(h4, batch)
        elif self.pooling == "mean":
            hg = global_mean_pool(h4, batch)
        elif self.pooling == "sum":
            hg = global_add_pool(h4, batch)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        # Final MLP
        out = self.fc1(hg)
        out = self.fc_norm(out)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        if self.output_dim == 1:
            out = torch.sigmoid(out)
        return out


class CustomDataset(Dataset):
    @staticmethod
    def preprocess(adj, features):
        """Transforms the adjacency matrix and features into tensors."""
        adj = torch.tensor(adj, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        return adj, features

    @staticmethod
    def process_graph(graph):
        adj, _, features = graph.get_adjacency_matrix()
        adj, features = CustomDataset.preprocess(adj, features)
        return graph.index, adj, features

    def __init__(self, models_dict_path, accuracies=None, use_tqdm=False):
        self.models_dict_path = models_dict_path

        self.accuracies = (
            torch.tensor(accuracies, dtype=torch.float)
            if accuracies is not None
            else None
        )

    def __getitem__(self, index):
        path = self.models_dict_path[index]
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        graph = Graph(data, index=index)
        _, adj, features = self.process_graph(graph)
        edge_index, _ = dense_to_sparse(adj)

        data = Data(x=features, edge_index=edge_index)
        data.index = index
        if self.accuracies is not None:
            data.y = self.accuracies[index]
        return data

    def __len__(self):
        return len(self.models_dict_path)


class TripletGraphDataset(Dataset):
    def __init__(self, base_dataset, diversity_matrix):
        """
        base_dataset: CustomDataset that transmits data from .index
        diversity_matrix: matrix [M, M], M >= N, value {1, -1, 0}
        """
        self.base = base_dataset
        self.div = diversity_matrix
        self.N = len(self.base)

        # Building a display of the original index -> internal
        # example: if base_dataset[5].index == 42, then orig2int[42] = 5
        self.orig2int = {self.base[i].index: i for i in range(self.N)}

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1) Get Data and its original index
        anchor = self.base[idx]
        anchor_orig = anchor.index  # in range [0, M-1]

        # 2) Get the row of diversity_matrix by the original index
        row = self.div[anchor_orig]  # length M

        # 3) Find original indices of positive and negative examples
        pos_orig = np.where((row == 1) & (np.arange(len(row)) != anchor_orig))[0]
        neg_orig = np.where(row == -1)[0]

        # 4) Filter by presence in self.orig2int
        pos_orig = [i for i in pos_orig if i in self.orig2int]
        neg_orig = [i for i in neg_orig if i in self.orig2int]

        # 5) Check for at least one positive and negative example
        if len(pos_orig) == 0 or len(neg_orig) == 0:
            raise IndexError(f"No valid pos/neg for original index {anchor_orig}")

        # 6) Randomly select appropriate indices
        pos_o = int(np.random.choice(pos_orig))
        neg_o = int(np.random.choice(neg_orig))

        # 7) Convert to internal indices and get Data
        pos_int = self.orig2int[pos_o]
        neg_int = self.orig2int[neg_o]

        positive = self.base[pos_int]
        negative = self.base[neg_int]

        # 8) Return three Data and a tensor of original indices
        idx_triplet = torch.tensor([anchor_orig, pos_o, neg_o], dtype=torch.long)
        return anchor, positive, negative, idx_triplet


def collate_triplets(batch):
    """
    batch: list of types (anchor, pos, neg, idx_triplet)
    is returned:
        - Three Batched Data
        - one LongTensor [batch_size, 3] with the original indexes
    """
    anchors, positives, negatives, idxs = zip(*batch)
    batch_anchor = Batch.from_data_list(anchors)
    batch_positive = Batch.from_data_list(positives)
    batch_negative = Batch.from_data_list(negatives)
    # assemble the matrix of indexes shape=(batch_size,3)
    idx_tensor = torch.cat(idxs, dim=0).view(-1, 3)
    return batch_anchor, batch_positive, batch_negative, idx_tensor


def collate_graphs(batch):
    """
    batch: list of torch_geometric.data.Data
    Returns Batch, which can be passed to GNN.
    """
    return Batch.from_data_list(batch)


def train_model_diversity(
    model,
    train_loader,  # DataLoader returns (anchor_batch, pos_batch, neg_batch, idx_triplet)
    valid_loader,  # The same for validation
    optimizer,
    criterion,
    num_epochs,
    device="cpu",
    developer_mode=False,
    final_lr=0.001,
    draw_figure=False,
):
    model.to(device)
    train_losses, valid_losses = [], []
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=final_lr)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # --------------------
        # 1) Training pass
        # --------------------
        model.train()
        running_loss = 0.0
        n_batches = 0

        for i, (anchor_batch, pos_batch, neg_batch, idx_triplet) in enumerate(
            train_loader
        ):
            if developer_mode and i > 0:
                break

            optimizer.zero_grad()
            # Move the entire batch to device
            anchor_batch = anchor_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            # Feed through the model
            emb_anchor = model(
                anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch
            )
            emb_pos = model(pos_batch.x, pos_batch.edge_index, pos_batch.batch)
            emb_neg = model(neg_batch.x, neg_batch.edge_index, neg_batch.batch)

            # Calculate loss, backward, step
            loss = criterion(emb_anchor, emb_pos, emb_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # --------------------
        # 2) Validation
        # --------------------
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for i, (anchor_batch, pos_batch, neg_batch, idx_triplet) in enumerate(
                valid_loader
            ):
                if developer_mode and i > 0:
                    break

                # Move the entire batch to device
                anchor_batch = anchor_batch.to(device)
                pos_batch = pos_batch.to(device)
                neg_batch = neg_batch.to(device)

                emb_anchor = model(
                    anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch
                )
                emb_pos = model(pos_batch.x, pos_batch.edge_index, pos_batch.batch)
                emb_neg = model(neg_batch.x, neg_batch.edge_index, neg_batch.batch)

                loss = criterion(emb_anchor, emb_pos, emb_neg)
                val_loss += loss.item()
                n_val_batches += 1

        avg_valid_loss = val_loss / max(1, n_val_batches)
        valid_losses.append(avg_valid_loss)

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1}/{num_epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}"
        )

    if draw_figure:
        plot_train_valid_losses(train_losses, valid_losses)

    return train_losses, valid_losses


def train_model_accuracy(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    num_epochs,
    device="cpu",
    developer_mode=False,
    final_lr=0.001,
    draw_figure=False,
):
    model.to(device)
    train_losses = []
    valid_losses = []

    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=final_lr)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        n_train_samples = 0

        for i, data in enumerate(train_loader):
            if developer_mode and i > 0:
                break

            data = data.to(device)
            optimizer.zero_grad()

            prediction = model(data.x, data.edge_index, data.batch).squeeze()
            target = data.y.float()

            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs  # весим loss по числу графов
            n_train_samples += data.num_graphs

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train_samples)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        n_val_samples = 0

        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                if developer_mode and i > 0:
                    break

                data = data.to(device)

                prediction = model(data.x, data.edge_index, data.batch).squeeze()
                target = data.y.float()

                loss = criterion(prediction, target)
                valid_loss += loss.item() * data.num_graphs
                n_val_samples += data.num_graphs

        avg_valid_loss = valid_loss / max(1, n_val_samples)
        valid_losses.append(avg_valid_loss)

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss * 1e4:.4f}, "
            f"Valid Loss: {avg_valid_loss * 1e4:.4f}, LR: {lr:.6f}"
        )

    if draw_figure:
        tmp_train_losses = np.array(train_losses)
        tmp_valid_losses = np.array(valid_losses) * 1e4
        plot_train_valid_losses(tmp_train_losses, tmp_valid_losses)

    return train_losses, valid_losses


def plot_train_valid_losses(train_losses, valid_losses):
    plt.figure(figsize=(12, 6))
    plt.rc("font", size=20)
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss"
    )
    plt.plot(
        range(1, len(valid_losses) + 1), valid_losses, marker="s", label="Valid Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_positive_and_negative(diversity_matrix, indices, dataset=None):
    positive_indices = []
    negative_indices = []

    for index in indices:
        positive = np.where(
            (diversity_matrix[index, :] == 1)
            & (np.arange(len(diversity_matrix)) != index)
        )[0].tolist()
        negative = np.where(diversity_matrix[index, :] == -1)[0].tolist()

        if dataset is not None:
            appropriate_indexes = [dataset[i][2] for i in range(len(dataset))]

            positive = [
                appropriate_indexes.index(idx)
                for idx in positive
                if idx in appropriate_indexes
            ]
            negative = [
                appropriate_indexes.index(idx)
                for idx in negative
                if idx in appropriate_indexes
            ]

        if not positive or not negative:
            print(f"Positive or negative samples are empty for index {index}!")
            positive_indices.append(None)
            negative_indices.append(None)
        else:
            # Выбираем случайный положительный и отрицательный пример
            positive_indices.append(np.random.choice(positive))
            negative_indices.append(np.random.choice(negative))

    return positive_indices, negative_indices


def extract_embeddings(model, data_loader, device, use_tqdm=True):
    model.to(device)
    model.eval()
    embeddings = []
    indices = []

    iterator = tqdm(data_loader) if use_tqdm else data_loader

    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = batch.to(device)

            batch_embeddings = model(batch.x, batch.edge_index, batch.batch)

            embeddings.append(batch_embeddings.cpu().numpy())
            indices.append(batch.index.cpu().numpy())

    embeddings = np.vstack(embeddings).squeeze()
    indices = np.concatenate(indices)
    return embeddings, indices
