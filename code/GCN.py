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
from joblib import Parallel, delayed
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

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
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = self.graph_norm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = F.relu(self.gc2(x, edge_index))
        x = self.graph_norm(x)
        x = self.layer_norm(x)
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
        self.hidden_dim = 64  # базовая размерность скрытого слоя

        self.gc1 = GCNConv(input_dim, self.hidden_dim)
        self.gc2 = GCNConv(self.hidden_dim, 256)
        self.gc3 = GCNConv(256, 512)
        self.gc4 = GCNConv(512, self.hidden_dim)

        self.residual_proj = (
            nn.Linear(input_dim, self.hidden_dim) if input_dim != self.hidden_dim else nn.Identity()
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
            x = torch.sum(x, dim=0)
        else:
            raise ValueError("Unsupported pooling method. Use 'max', 'mean' or 'sum'.")

        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        if self.output_dim == 1:
            x = torch.sigmoid(x)
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
    @staticmethod
    def preprocess(adj, features):
        """Преобразует матрицу смежности и признаки в тензоры."""
        adj = torch.tensor(adj, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        return adj, features

    @staticmethod
    def process_graph(graph):
        adj, _, features = graph.get_adjacency_matrix()
        adj, features = CustomDataset.preprocess(adj, features)
        return graph.index, adj, features

    def __init__(self, graphs, accuracies=None, use_tqdm=False):
        self.indexes = []
        self.adjs, self.features = [], []
        self.accuracies = torch.tensor(accuracies, dtype=torch.float) if accuracies is not None else None

        iterator = tqdm(graphs) if use_tqdm else graphs

        results = Parallel(n_jobs=-1)(
            delayed(CustomDataset.process_graph)(graph) for graph in iterator
        )

        for index, adj, features in results:
            self.indexes.append(index)
            self.adjs.append(adj)
            self.features.append(features)

    def __getitem__(self, index):
        adj = self.adjs[index]
        features = self.features[index]
        edge_index, _ = dense_to_sparse(adj)

        data = Data(x=features, edge_index=edge_index)
        data.index = self.indexes[index]
        if self.accuracies is not None:
            data.y = self.accuracies[index]
        return data

    def __len__(self):
        return len(self.indexes)

class TripletGraphDataset(Dataset):
    def __init__(self, base_dataset, diversity_matrix):
        """
        base_dataset: CustomDataset, отдающий Data с .index
        diversity_matrix: матрица [M, M], M >= N, значений {1, -1, 0}
        """
        self.base = base_dataset
        self.div = diversity_matrix
        self.N = len(self.base)

        # Строим отображение оригинального индекса -> внутренний
        # пример: если base_dataset[5].index == 42, то orig2int[42] = 5
        self.orig2int = {
            self.base[i].index: i
            for i in range(self.N)
        }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1) Получаем Data и его оригинальный индекс
        anchor = self.base[idx]
        anchor_orig = anchor.index  # в диапазоне [0, M-1]

        # 2) Берём строку diversity_matrix по оригинальному индексу
        row = self.div[anchor_orig]  # длина M

        # 3) Находим оригинальные индексы положительных и отрицательных
        pos_orig = np.where((row == 1) & (np.arange(len(row)) != anchor_orig))[0]
        neg_orig = np.where(row == -1)[0]

        # 4) Фильтруем по наличию в self.orig2int
        pos_orig = [i for i in pos_orig if i in self.orig2int]
        neg_orig = [i for i in neg_orig if i in self.orig2int]

        # 5) Проверка наличия хотя бы одного положительного и отрицательного
        if len(pos_orig) == 0 or len(neg_orig) == 0:
            raise IndexError(f"No valid pos/neg for original index {anchor_orig}")

        # 6) Случайно выбираем подходящие индексы
        pos_o = int(np.random.choice(pos_orig))
        neg_o = int(np.random.choice(neg_orig))

        # 7) Переводим в внутренние индексы и получаем Data
        pos_int = self.orig2int[pos_o]
        neg_int = self.orig2int[neg_o]

        positive = self.base[pos_int]
        negative = self.base[neg_int]

        # 8) Возвращаем три Data и тензор оригинальных индексов
        idx_triplet = torch.tensor([anchor_orig, pos_o, neg_o], dtype=torch.long)
        return anchor, positive, negative, idx_triplet

def collate_triplets(batch):
    """
    batch: list of tuples (anchor, pos, neg, idx_triplet)
    Возвращаем:
      - три Batched Data
      - один LongTensor [batch_size, 3] с исходными индексами
    """
    anchors, positives, negatives, idxs = zip(*batch)
    batch_anchor = Batch.from_data_list(anchors)
    batch_positive = Batch.from_data_list(positives)
    batch_negative = Batch.from_data_list(negatives)
    # соберём матрицу индексов shape=(batch_size,3)
    idx_tensor = torch.cat(idxs, dim=0).view(-1, 3)
    return batch_anchor, batch_positive, batch_negative, idx_tensor


def collate_graphs(batch):
    """
    batch: list of torch_geometric.data.Data
    Возвращает Batch, который можно подать в GNN.
    """
    return Batch.from_data_list(batch)


def train_model_diversity(
    model,
    train_loader,         # DataLoader выдаёт (anchor_batch, pos_batch, neg_batch)
    valid_loader,         # То же для валидации
    optimizer,
    criterion,
    num_epochs,
    device="cpu",
    developer_mode=False,
):
    model.to(device)
    train_losses, valid_losses = [], []
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # --------------------
        # 1) Тренировочный проход
        # --------------------
        model.train()
        running_loss = 0.0
        n_batches = 0

        for i, (anchor_batch, pos_batch, neg_batch, idx_triplet) in enumerate(train_loader):
            if developer_mode and i > 0:
                break

            optimizer.zero_grad()
            # Переносим весь батч на device
            anchor_batch = anchor_batch.to(device)
            pos_batch    = pos_batch.to(device)
            neg_batch    = neg_batch.to(device)

            # Прогоняем через модель
            emb_anchor = model(anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch)
            emb_pos    = model(pos_batch.x,    pos_batch.edge_index,    pos_batch.batch)
            emb_neg    = model(neg_batch.x,    neg_batch.edge_index,    neg_batch.batch)

            # Считаем loss, backward, step
            loss = criterion(emb_anchor, emb_pos, emb_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # --------------------
        # 2) Валидация
        # --------------------
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for i, (anchor_batch, pos_batch, neg_batch, idx_triplet) in enumerate(valid_loader):
                if developer_mode and i > 0:
                    break

                # перенос на device
                anchor_batch = anchor_batch.to(device)
                pos_batch    = pos_batch.to(device)
                neg_batch    = neg_batch.to(device)

                emb_anchor = model(anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch)
                emb_pos    = model(pos_batch.x,    pos_batch.edge_index,    pos_batch.batch)
                emb_neg    = model(neg_batch.x,    neg_batch.edge_index,    neg_batch.batch)

                loss = criterion(emb_anchor, emb_pos, emb_neg)
                val_loss += loss.item()
                n_val_batches += 1

        avg_valid_loss = val_loss / max(1, n_val_batches)
        valid_losses.append(avg_valid_loss)

        # --------------------
        # 3) Лог и визуализация
        # --------------------
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
        plt.plot(range(1, len(valid_losses)+1), valid_losses, marker='s', label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}")

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
        n_train_batches = 0

        for i, data in enumerate(train_loader):
            if developer_mode and i > 0:
                break

            data = data.to(device)
            optimizer.zero_grad()

            # Явно передаем `edge_index` и `x`
            prediction = model(data.x, data.edge_index, data.batch).squeeze()  
            target = data.y.float()

            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                if developer_mode and i > 0:
                    break

                data = data.to(device)

                prediction = model(data.x, data.edge_index, data.batch).squeeze()
                target = data.y.float()

                loss = criterion(prediction, target)
                valid_loss += loss.item()
                n_val_batches += 1

        avg_valid_loss = valid_loss / max(1, n_val_batches)
        valid_losses.append(avg_valid_loss)

        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except:
            pass

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss")
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker="o", label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.ylim(0, 0.002)
        plt.grid(True)
        plt.legend()
        plt.show()

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, "
              f"Valid Loss: {avg_valid_loss:.4f}, LR: {lr:.6f}")

    return train_losses, valid_losses

def get_positive_and_negative(diversity_matrix, indices, dataset=None):
    positive_indices = []
    negative_indices = []

    for index in indices:
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