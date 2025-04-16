import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
    GraphNorm,
)
from torch_geometric.utils import dense_to_sparse
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class SimpleGCN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, embedding_dim, dropout=0.5, pooling="max"
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

        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.graph_norm = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = self.graph_norm(x)
        x = self.dropout(x)

        x = F.relu(self.gc2(x, edge_index))
        x = self.graph_norm(x)
        x = self.dropout(x)

        # Глобальное агрегирование узловых признаков для получения представления всего графа
        if self.pooling == "max":
            pooled = torch.max(x, dim=0).values
        elif self.pooling == "mean":
            pooled = torch.mean(x, dim=0)
        elif self.pooling == "sum":
            pooled = torch.sum(x, dim=0)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' или 'sum'.")

        # Преобразование агрегированного представления в эмбеддинг графа
        embedding = self.fc(pooled)

        if self.embedding_dim == 1:
            embedding = nn.Sigmoid()(embedding)
        return embedding


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim=16, dropout=0.5, pooling="max"):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gc1 = GCNConv(input_dim, 128)
        self.gc2 = GCNConv(128, 256)
        self.gc3 = GCNConv(256, 64)
        self.layer_norm = nn.LayerNorm(64)
        self.fc = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        self.key = nn.Linear(256, 256)
        self.query = nn.Linear(256, 256)

        self.residual_proj = (
            nn.Linear(input_dim, 64) if input_dim != 64 else nn.Identity()
        )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x)

        x = F.leaky_relu(self.gc1(x, edge_index))
        x = self.dropout(x)

        x = F.leaky_relu(self.gc2(x, edge_index))
        x = self.dropout(x)

        keys = self.key(x)  # [N, 128]
        queries = self.query(x)  # [N, 128]
        attn_scores = torch.mm(queries, keys.T)  # [N, N]

        row, col = edge_index
        mask = torch.zeros_like(attn_scores)
        mask[row, col] = 1
        attn_scores = attn_scores * mask
        attn_scores = F.softmax(attn_scores, dim=-1)

        x = torch.mm(attn_scores, x)  # [N, N] * [N, 128] → [N, 128]

        x = F.leaky_relu(self.gc3(x, edge_index))
        x = self.dropout(x)

        x = self.layer_norm(x + residual)  # LayerNorm

        if self.pooling == "max":
            x = torch.max(x, dim=0).values
        elif self.pooling == "mean":
            x = torch.mean(x, dim=0)
        elif self.pooling == "sum":
            x = torch.sum(x, dim=0)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        x = self.fc(x)
        if self.output_dim == 1:
            x = nn.Sigmoid()(x)
        return x


class SimpleGAT(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        out_dim,
        dropout=0.5,
        pooling="max",
        heads=1,
    ):
        """
        input_dim: размер one-hot признаков узлов
        embed_dim: размер обучаемого эмбеддинга узлов
        hidden_dim: скрытое пространство для GAT
        out_dim: размер итогового графового эмбеддинга
        pooling: 'max', 'mean', или 'sum'
        """
        super(SimpleGAT, self).__init__()

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        self.node_encoder = nn.Linear(
            input_dim, embed_dim
        )  # обучаемый слой проекции one-hot -> dense

        self.gat1 = GATConv(embed_dim, hidden_dim, heads=heads, concat=True)
        self.norm1 = GraphNorm(hidden_dim * heads)

        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.norm2 = GraphNorm(hidden_dim * heads)

        self.fc = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.node_encoder(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x)
        x = self.dropout(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.norm2(x)
        x = self.dropout(x)

        # Пуллинг по графу (поддержка батча)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.pooling == "max":
            pooled = global_max_pool(x, batch)
        elif self.pooling == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            pooled = global_add_pool(x, batch)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        embedding = self.fc(pooled)

        if self.fc.out_features == 1:
            embedding = torch.sigmoid(embedding)

        return embedding


class CustomDataset(Dataset):
    def __init__(self, graphs, accuracies=None):
        self.indexes = []
        self.adjs, self.features = [], []
        if accuracies is None:
            self.accuracies = None
        else:
            self.accuracies = torch.tensor(accuracies, dtype=torch.float)

        for graph in tqdm(graphs):
            adj, _, features = graph.get_adjacency_matrix()
            adj, features = self.preprocess(adj, features)

            self.adjs.append(adj)
            self.features.append(features)
            self.indexes.append(graph.index)

    def preprocess(self, adj, features):
        """Преобразует матрицу смежности и признаки в тензоры."""
        adj = torch.tensor(adj, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        return adj, features

    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray)):
            result = []
            for i in index:
                if self.accuracies is None:
                    result.append([self.adjs[i], self.features[i], self.indexes[i]])
                else:
                    result.append([self.adjs[i], self.features[i], self.indexes[i], self.accuracies[i]])
            return result
        else:
            if self.accuracies is None:
                return self.adjs[index], self.features[index], self.indexes[index]
            return self.adjs[index], self.features[index], self.indexes[index], self.accuracies[index]

    def __len__(self):
        return len(self.indexes)

def train_model_diversity(
    model,
    train_loader,
    valid_loader,
    discrete_diversity_matrix,
    optimizer,
    criterion,
    num_epochs,
    device="cpu",
    developer_mode=False,
):
    model.to(device)  # Перемещение модели на GPU
    train_losses = []
    valid_losses = []

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        for i, (adj, features, index, _) in enumerate(train_loader):
            if developer_mode and i > 0:
                break  # Only process one batch in developer mode

            optimizer.zero_grad()

            # Перемещение на GPU
            adj, features = adj.to(device), features.to(device)
            edge_index, _ = dense_to_sparse(adj)  # Преобразование adj в edge_index

            anchor = model(features, edge_index)

            positive_index, negative_index = get_positive_and_negative(
                discrete_diversity_matrix, index, train_loader.dataset
            )
            if positive_index is None or negative_index is None:
                continue

            positive_adj, positive_feat, *_ = train_loader.dataset[positive_index]
            negative_adj, negative_feat, *_ = train_loader.dataset[negative_index]

            # Перемещение положительных и отрицательных примеров на GPU
            positive_adj, positive_feat = positive_adj.to(device), positive_feat.to(
                device
            )
            negative_adj, negative_feat = negative_adj.to(device), negative_feat.to(
                device
            )

            positive_edge_index, _ = dense_to_sparse(positive_adj)
            negative_edge_index, _ = dense_to_sparse(negative_adj)

            positive = model(positive_feat, positive_edge_index)
            negative = model(negative_feat, negative_edge_index)

            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Оценка на валидации
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (adj, features, index, _) in enumerate(valid_loader):
                if developer_mode and i > 0:
                    break  # Only process one batch in developer mode

                adj, features = adj.to(device), features.to(device)
                edge_index, _ = dense_to_sparse(adj)  # Преобразование adj в edge_index

                anchor = model(features, edge_index)

                positive_index, negative_index = get_positive_and_negative(
                    discrete_diversity_matrix, index, valid_loader.dataset
                )
                if positive_index is None or negative_index is None:
                    continue

                positive_adj, positive_feat, *_ = valid_loader.dataset[positive_index]
                negative_adj, negative_feat, *_ = valid_loader.dataset[negative_index]

                positive_adj, positive_feat = positive_adj.to(device), positive_feat.to(
                    device
                )
                negative_adj, negative_feat = negative_adj.to(device), negative_feat.to(
                    device
                )

                positive_edge_index, _ = dense_to_sparse(positive_adj)
                negative_edge_index, _ = dense_to_sparse(negative_adj)

                positive = model(positive_feat, positive_edge_index)
                negative = model(negative_feat, negative_edge_index)

                loss = criterion(anchor, positive, negative)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except:
            pass

        plt.figure(figsize=(12, 6))

        plt.rc('font', size=16)          # controls default text sizes
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=16)    # legend fontsize
        plt.rc('figure', titlesize=16)   # fontsize of the figure title

        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            marker="o",
            label="Train Loss",
        )
        plt.plot(
            range(1, len(valid_losses) + 1),
            valid_losses,
            marker="o",
            label="Valid Loss",
        )
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
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
):
    model.to(device)
    train_losses = []
    valid_losses = []

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0

        for i, (adj, features, _, accuracy) in enumerate(train_loader):
            if developer_mode and i > 0:
                break  # Only process one batch in developer mode

            optimizer.zero_grad()

            adj = adj.to(device)
            features = features.to(device)
            accuracy = accuracy.to(device).float()

            edge_index, _ = dense_to_sparse(adj)
            prediction = model(features, edge_index).squeeze()

            loss = criterion(prediction, accuracy)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (adj, features, _, accuracy) in enumerate(valid_loader):
                if developer_mode and i > 0:
                    break  # Only process one batch in developer mode

                adj = adj.to(device)
                features = features.to(device)
                accuracy = accuracy.to(device).float()

                edge_index, _ = dense_to_sparse(adj)
                prediction = model(features, edge_index).squeeze()

                loss = criterion(prediction, accuracy)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except:
            pass

        plt.figure(figsize=(12, 6))

        plt.rc('font', size=16)          # controls default text sizes
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=16)    # legend fontsize
        plt.rc('figure', titlesize=16)   # fontsize of the figure title

        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            marker="o",
            label="Train Loss",
        )
        plt.plot(
            range(1, len(valid_losses) + 1),
            valid_losses,
            marker="o",
            label="Valid Loss",
        )
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0, 0.002)
        plt.grid(True)
        plt.legend()
        plt.show()

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    return train_losses, valid_losses


def get_positive_and_negative(diversity_matrix, index, dataset=None):
    positive = np.where(
        (diversity_matrix[index, :] == 1) & (np.arange(len(diversity_matrix)) != index)
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
        print("Both positive and negative samples are empty!")
        return None, None

    return np.random.choice(positive), np.random.choice(negative)
