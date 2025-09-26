import sys
import os
import json
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
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
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Graph import Graph


class GATBlock(nn.Module):
    """
    - Identity residual, если in_dim == out_dim
    - Поддержка Pre-Norm (до GAT) и Post-Norm (после)
    - ELU по умолчанию
    - DropEdge (edge_dropout > 0)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.5,
        edge_dropout: float = 0.0,
        pre_norm: bool = False,
        activation: nn.Module = None,
    ):
        super().__init__()
        assert out_dim % heads == 0, "out_dim должно быть кратно числу голов."

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pre_norm = pre_norm
        self.edge_dropout = float(edge_dropout)

        self.gat = GATv2Conv(in_dim, out_dim // heads, heads=heads)
        self.res_proj = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.act = activation if activation is not None else nn.ELU()

        # === Исправление: нормализация зависит от режима ===
        if pre_norm:
            # Pre-Norm: нормализуем вход → используем in_dim
            self.norm_pre = GraphNorm(in_dim)
            self.norm_post = nn.Identity()  # не нужна после
        else:
            # Post-Norm: нормализуем выход → используем out_dim
            self.norm_pre = nn.Identity()
            self.norm_post = GraphNorm(out_dim)

    def reset_parameters(self):
        self.gat.reset_parameters()
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)
            if self.res_proj.bias is not None:
                nn.init.zeros_(self.res_proj.bias)
        if hasattr(self, "norm_post") and isinstance(self.norm_post, GraphNorm):
            self.norm_post.reset_parameters()
        if hasattr(self, "norm_pre") and isinstance(self.norm_pre, GraphNorm):
            self.norm_pre.reset_parameters()

    def forward(self, x, edge_index, batch):
        # Edge dropout (только в train)
        if self.training and self.edge_dropout > 0.0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)

        # === Pre-Norm: до GAT, по in_dim ===
        x_for_gat = (
            self.norm_pre(x, batch) if isinstance(self.norm_pre, GraphNorm) else x
        )

        h = self.gat(x_for_gat, edge_index)
        res = self.res_proj(x)
        h = self.act(h + res)

        # === Post-Norm: после, по out_dim ===
        if isinstance(self.norm_post, GraphNorm):
            h = self.norm_post(h, batch)

        h = self.dropout(h)
        return h


class GAT_ver_1(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        dropout=0.5,
        pooling="mean",
        heads=4,
        output_activation="none",
    ):
        super().__init__()
        self.hidden_dim = 64
        self.output_dim = output_dim

        self.block1 = GATBlock(input_dim, self.hidden_dim, heads=heads, dropout=dropout)
        self.block2 = GATBlock(self.hidden_dim, 256, heads=heads, dropout=dropout)
        self.block3 = GATBlock(256, 256, heads=heads, dropout=dropout)
        self.block4 = GATBlock(256, self.hidden_dim, heads=heads, dropout=dropout)

        self.pooling = pooling

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_norm = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

        self.output_activation = output_activation

    def forward(self, x, edge_index, batch=None):
        h1 = self.block1(x, edge_index, batch)
        h2 = self.block2(h1, edge_index, batch)
        h3 = self.block3(h2, edge_index, batch)
        h4 = self.block4(h3, edge_index, batch)

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

        if self.output_activation == "sigmoid" and self.output_dim == 1:
            return torch.sigmoid(out)
        elif self.output_activation == "softmax":
            return F.log_softmax(out, dim=-1)
        elif self.output_activation == "l2":
            return F.normalize(out, p=2, dim=-1)
        else:
            return out


class GAT_ver_2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        dropout: float = 0.5,
        heads: int = 4,
        output_activation: str = "none",
        pooling: str = "attn",
        edge_dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = 64
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.pooling = pooling

        act = nn.ELU()

        self.block1 = GATBlock(
            input_dim,
            self.hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            pre_norm=pre_norm,
            activation=act,
        )
        self.block2 = GATBlock(
            self.hidden_dim,
            256,
            heads=heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            pre_norm=pre_norm,
            activation=act,
        )
        self.block3 = GATBlock(
            256,
            256,
            heads=heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            pre_norm=pre_norm,
            activation=act,
        )
        self.block4 = GATBlock(
            256,
            self.hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            pre_norm=pre_norm,
            activation=act,
        )

        def make_attn_pool(d):
            gate_nn = nn.Sequential(
                nn.Linear(d, max(1, d // 2)),
                nn.ReLU(),
                nn.Linear(max(1, d // 2), 1),
            )
            return AttentionalAggregation(gate_nn)

        self.attn1 = make_attn_pool(self.hidden_dim)  # 64
        self.attn2 = make_attn_pool(256)
        self.attn3 = make_attn_pool(256)
        self.attn4 = make_attn_pool(self.hidden_dim)  # 64

        self.jk_proj = nn.Linear(
            self.hidden_dim + 256 + 256 + self.hidden_dim, self.hidden_dim
        )

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_norm = nn.LayerNorm(self.hidden_dim)
        self.fc_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        h1 = self.block1(x, edge_index, batch)
        h2 = self.block2(h1, edge_index, batch)
        h3 = self.block3(h2, edge_index, batch)
        h4 = self.block4(h3, edge_index, batch)

        if self.pooling == "attn":
            p1 = self.attn1(h1, batch)
            p2 = self.attn2(h2, batch)
            p3 = self.attn3(h3, batch)
            p4 = self.attn4(h4, batch)
        elif self.pooling in {"max", "mean", "sum"}:
            from torch_geometric.nn import (
                global_max_pool,
                global_mean_pool,
                global_add_pool,
            )

            pool_fn = {
                "max": global_max_pool,
                "mean": global_mean_pool,
                "sum": global_add_pool,
            }[self.pooling]
            p1 = pool_fn(h1, batch)
            p2 = pool_fn(h2, batch)
            p3 = pool_fn(h3, batch)
            p4 = pool_fn(h4, batch)
        else:
            raise ValueError(
                "Unsupported pooling method. Use 'attn', 'max', 'mean' or 'sum'."
            )

        hg = torch.cat([p1, p2, p3, p4], dim=-1)
        hg = self.jk_proj(hg)

        out = self.fc1(hg)
        out = self.fc_norm(out)
        out = F.elu(out)
        out = self.fc_drop(out)
        out = self.fc2(out)

        if self.output_activation == "sigmoid" and self.output_dim == 1:
            return torch.sigmoid(out)
        elif self.output_activation == "softmax":
            return F.log_softmax(out, dim=-1)
        elif self.output_activation == "l2":
            return F.normalize(out, p=2, dim=-1)
        else:
            return out


# ВАЖНО: функция должна быть в глобальной области, чтобы multiprocessing мог её сериализовать
def load_single_graph(args: tuple):
    json_path_str, idx, accuracies = args
    try:
        json_path = Path(json_path_str)

        model_dict = json.loads(json_path.read_text(encoding="utf-8"))

        graph = Graph(model_dict, index=idx)
        adj, _, features = (
            graph.get_adjacency_matrix()
        )  # предполагаем: adj и features — list или np.array

        adj = np.array(adj)
        features = np.array(features)

        edge_index = np.stack(np.nonzero(adj)).tolist()

        result = {
            "idx": idx,
            "x": features.astype(
                np.float32
            ).tolist(),  # или .tolist() для JSON-совместимости
            "edge_index": edge_index,
            "y": float(accuracies[idx]) if accuracies is not None else None,
        }
        return result
    except Exception as e:
        print(f"Error loading {json_path_str}: {e}")
        return None


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
            model_dict = json.load(f)
        graph = Graph(model_dict, index=index)
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
    save_path="checkpoints/best_diversity_model.pth",  # путь к чекпоинту
):
    model.to(device)
    train_losses, valid_losses = [], []
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=final_lr)

    # --- Создаём временную папку ---
    checkpoint_dir = os.path.dirname(save_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        temp_dir_created = True
    else:
        temp_dir_created = False

    best_valid_loss = float("inf")

    try:
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

                # Move to device
                anchor_batch = anchor_batch.to(device)
                pos_batch = pos_batch.to(device)
                neg_batch = neg_batch.to(device)

                # Forward pass
                emb_anchor = model(
                    anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch
                )
                emb_pos = model(pos_batch.x, pos_batch.edge_index, pos_batch.batch)
                emb_neg = model(neg_batch.x, neg_batch.edge_index, neg_batch.batch)

                # Loss & step
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

            # === Сохранение лучшей модели ===
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), save_path)
                print(
                    f"✅ Best diversity model saved to {save_path} (Valid Loss: {avg_valid_loss:.4f})"
                )

            # === Логирование ===
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{num_epochs} — "
                f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}"
            )

        # === Загружаем лучшую модель в память ===
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"✅ Loaded best diversity model from {save_path}")

    except Exception as e:
        # Удаляем папку при ошибке
        if temp_dir_created and checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print(f"🧹 Temporary directory '{checkpoint_dir}' removed after error.")
        raise

    finally:
        # === Удаляем временную папку в любом случае ===
        if temp_dir_created and checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print(f"🧹 Temporary directory '{checkpoint_dir}' removed.")

    # === Построение графика ===
    tmp_train_losses = np.array(train_losses)
    tmp_valid_losses = np.array(valid_losses)
    plot_train_valid_losses(
        tmp_train_losses, tmp_valid_losses, file_name="diversity_model.png"
    )

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
    save_path="checkpoints/best_accuracy_model.pth",  # можно передать другой путь
):
    model.to(device)
    train_losses = []
    valid_losses = []

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=final_lr)

    # --- Создаём временную папку ---
    checkpoint_dir = os.path.dirname(save_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        temp_dir_created = True
    else:
        temp_dir_created = False

    best_valid_loss = float("inf")

    try:
        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            # === Training ===
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

                train_loss += loss.item() * data.num_graphs
                n_train_samples += data.num_graphs

            scheduler.step()
            avg_train_loss = train_loss / max(1, n_train_samples)
            train_losses.append(avg_train_loss)

            # === Validation ===
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

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), save_path)
                print(
                    f"✅ Best model saved to {save_path} (Valid Loss: {avg_valid_loss * 1e4:.4f})"
                )

            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}, Train Loss: {avg_train_loss * 1e4:.4f}, "
                f"Valid Loss: {avg_valid_loss * 1e4:.4f}, LR: {lr:.6f}"
            )

        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"✅ Loaded best model from {save_path}")

    except Exception as e:
        if temp_dir_created and checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print(f"🧹 Temporary directory '{checkpoint_dir}' removed after error.")
        raise

    finally:
        if temp_dir_created and checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print(f"🧹 Temporary directory '{checkpoint_dir}' removed.")

    tmp_train_losses = np.sqrt(np.array(train_losses))
    tmp_valid_losses = np.sqrt(np.array(valid_losses))
    plot_train_valid_losses(
        tmp_train_losses, tmp_valid_losses, file_name="accuracy_model.png"
    )

    save_accuracy_predictions(
        model=model,
        data_loader=valid_loader,
        device=device,
        file_path="logs/accuracy_predictions.txt",
        developer_mode=developer_mode,
    )

    return train_losses, valid_losses


def save_accuracy_predictions(
    model,
    data_loader,
    device="cpu",
    file_path="logs/accuracy_predictions.txt",
    developer_mode=False,
):
    model.eval()
    true_accs = []
    pred_accs = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if developer_mode and i > 0:
                break

            data = data.to(device)
            prediction = (
                model(data.x, data.edge_index, data.batch).squeeze().cpu().numpy()
            )
            target = data.y.cpu().numpy()

            # Переводим в проценты, если нужно
            if target.max() <= 1.0:
                target = target * 100  # [0, 1] → [0, 100]
            if prediction.max() <= 1.0:
                prediction = prediction * 100

            true_accs.extend(target)
            pred_accs.extend(prediction)

    true_accs = np.array(true_accs)
    pred_accs = np.array(pred_accs)

    # === Вычисляем метрики ДО сортировки ===
    r2 = r2_score(true_accs, pred_accs)
    mae = mean_absolute_error(true_accs, pred_accs)
    rmse = np.sqrt(mean_squared_error(true_accs, pred_accs))

    # === Сортируем по убыванию true_acc ===
    sorted_indices = np.argsort(true_accs)[::-1]  # [::-1] — это и есть убывание
    true_accs_sorted = true_accs[sorted_indices]
    pred_accs_sorted = pred_accs[sorted_indices]

    # === Сохраняем: сначала метрики, потом отсортированные данные ===
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# Accuracy prediction quality metrics\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"# Number of samples: {len(true_accs)}\n")
        f.write(f"\n# true_acc pred_acc (sorted by true_acc descending)\n")
        for true, pred in zip(true_accs_sorted, pred_accs_sorted):
            f.write(f"{true:.4f} {pred:.4f}\n")

    print(f"✅ Saved sorted predictions and metrics to {file_path}")
    print(f"   R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    print(f"   Entries sorted by true_acc (descending)")

    return true_accs_sorted, pred_accs_sorted


def plot_train_valid_losses(
    train_losses, valid_losses, file_name="train_valid_losses.png"
):
    os.makedirs("logs", exist_ok=True)
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
    plt.ylim(None, np.median(valid_losses) * 2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/" + file_name)
    plt.close()


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
