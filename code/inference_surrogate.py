import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import argparse
import shutil
from pathlib import Path

from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import torch
import gc
from torch.utils.data import DataLoader
from collections import deque
import shutil

# Custom imports
import sys

sys.path.insert(1, "../dependencies")

from dependencies.GCN import GAT, CustomDataset, collate_graphs, extract_embeddings
from dependencies.Graph import Graph
from dependencies.data_generator import generate_arch_dicts
from dependencies.train_config import TrainConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST = False


class InferSurrogate:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(self.config.device)
        self.dataset_path = Path(self.config.surrogate_inference_path)

    def initialize_models(self):
        self.config.model_accuracy = GAT(
            input_dim=self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
        ).to(self.device)
        state_dict = torch.load(
            self.dataset_path / "model_accuracy.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.config.model_accuracy.load_state_dict(state_dict)
        self.config.model_accuracy.eval()

        self.config.model_diversity = GAT(
            input_dim=self.config.input_dim,
            output_dim=self.config.div_output_dim,
            dropout=self.config.div_dropout,
            heads=self.config.div_n_heads,
        ).to(self.device)
        state_dict = torch.load(
            self.dataset_path / "model_diversity.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.config.model_diversity.load_state_dict(state_dict)
        self.config.model_diversity.eval()

    def _find_dist_to_best(self, best_embs, emb):
        """
        best_embs: tensor [k, d], emb: tensor [d] или [1, d]
        возвращает min расстояние от emb до любого из best_embs
        """
        if best_embs.numel() == 0:
            return float("inf")
        dists = torch.cdist(emb.unsqueeze(0), best_embs, p=2)  # [1, k]
        return dists.min().item()

    def architecture_search(self):
        tmp_dir = Path(self.config.tmp_archs_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # while len(self.config.best_models) < self.config.n_ensemble_models:
        #     print(
        #         f"\nProgress: {len(self.config.best_models)}/{self.config.n_ensemble_models} selected, pool size {len(self.config.potential_archs)}/{self.config.n_models_in_pool}"
        #     )

        # 1) Сгенерировать архитектуры
        arch_dicts = generate_arch_dicts(
            self.config.n_models_to_generate, use_tqdm=True
        )

        # 2) Сохраняем как JSON-файлы
        arch_paths = []
        for i, arch in enumerate(arch_dicts):
            path = tmp_dir / f"arch_{len(os.listdir(tmp_dir)) + i}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(arch, f)
            arch_paths.append(path)

        # 3) Датасет и DataLoader
        dataset = CustomDataset(arch_paths, use_tqdm=True)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_inference,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
        )

        # 4) Извлечь эмбеддинги
        with torch.no_grad():
            emb_acc_np, _ = extract_embeddings(
                self.config.model_accuracy, loader, device, use_tqdm=True
            )
            emb_div_np, _ = extract_embeddings(
                self.config.model_diversity, loader, device, use_tqdm=True
            )

        # 5) Фильтрация по точности
        mask = emb_acc_np >= self.config.min_accuracy_for_pool
        valid_paths = [p for p, ok in zip(arch_paths, mask) if ok]
        valid_div_embs = emb_div_np[mask].astype(np.float16)
        valid_accs = emb_acc_np[mask]

        # 6) Добавляем в пул
        for path, emb, acc in zip(valid_paths, valid_div_embs, valid_accs):
            arch = json.loads(path.read_text(encoding="utf-8"))
            self.config.potential_archs.append(arch)
            self.config.potential_embeddings.append(emb)
            self.config.potential_accuracies.append(acc)

        # 7) Удаляем все файлы, которые не попали в пул
        kept = set(valid_paths)
        for path in arch_paths:
            if path not in kept and path.exists():
                path.unlink()

        # 8) Очистка мусора
        del arch_dicts, arch_paths, dataset, loader
        del emb_acc_np, emb_div_np
        torch.cuda.empty_cache()
        gc.collect()

            # # 9) Если пул заполнен — выбираем наиболее разнообразные модели
            # while (
            #     len(self.config.potential_archs) >= self.config.n_models_in_pool
            #     and len(self.config.best_models) < self.config.n_ensemble_models
            # ):
            #     if self.config.best_embeddings:
            #         best_arr = np.stack(self.config.best_embeddings)
            #         distances = [
            #             np.min(np.linalg.norm(emb - best_arr, axis=1))
            #             for emb in self.config.potential_embeddings
            #         ]
            #     else:
            #         distances = [np.inf] * len(self.config.potential_embeddings)

            #     farthest = int(np.argmax(distances))

            #     self.config.best_models.append(
            #         self.config.potential_archs.pop(farthest)
            #     )
            #     self.config.best_embeddings.append(
            #         self.config.potential_embeddings.pop(farthest)
            #     )
            #     acc = self.config.potential_accuracies.pop(farthest)
            #     print(
            #         f"Selected #{len(self.config.best_models)}/{self.config.n_ensemble_models}: acc={acc:.4f}, dist={distances[farthest]:.4f}"
            #     )

        # В конце удалим временные файлы полностью
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def select_central_models_by_clusters(self):
        """
        Кластеризует potential_embeddings на K кластеров,
        и из каждого выбирает модель, ближайшую к центроиду.

        Если plot_tsne=True, сначала уменьшает размерность эмбеддингов
        до 50 с помощью PCA, затем проецирует вместе с центроидами
        на плоскость t-SNE и рисует результат.

        Возвращает:
            selected_archs: список выбранных архитектур (K штук)
            selected_embs: соответствующие эмбеддинги
            selected_accs: соответствующие точности
        """
        # Приводим к numpy
        X = np.array(self.config.potential_embeddings, dtype=np.float32)
        accs = np.array(self.config.potential_accuracies, dtype=np.float32)
        
        # if len(X) > self.config.n_models_in_pool:
        #     X = X[:self.config.n_models_in_pool]
        #     accs = accs[:self.config.n_models_in_pool]

        # 1. Кластеризация на оригинальных эмбеддингах
        kmeans = KMeans(
            n_clusters=self.config.n_ensemble_models,
            random_state=self.config.seed,
            n_init=10,
        )
        cluster_ids = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        for cluster_id in range(self.config.n_ensemble_models):
            idxs = np.where(cluster_ids == cluster_id)[0]
            if idxs.size == 0:
                continue

            cluster_X = X[idxs]
            centroid = centroids[cluster_id]

            dists = np.linalg.norm(cluster_X - centroid, axis=1)
            local_best = np.argmin(dists)
            best_global = idxs[local_best]

            self.config.selected_archs.append(self.config.potential_archs[best_global])
            self.config.selected_embs.append(X[best_global])
            self.config.selected_accs.append(accs[best_global])
            self.config.selected_indices.append(best_global)
        
        self.paint_tsne(X, centroids, cluster_ids)
    
    def random_choice_out_of_best(self):
        X = np.array(self.config.potential_embeddings, dtype=np.float32)
        accs = np.array(self.config.potential_accuracies, dtype=np.float32)

        print(X.shape)
        raise "KUKU"

        for i in range(self.config.n_ensemble_models):
            chosen = np.random.randint()
            self.config.selected_archs.append(self.config.potential_archs[best_global])
            self.config.selected_embs.append(X[best_global])
            self.config.selected_accs.append(accs[best_global])
            self.config.selected_indices.append(best_global)


    
    def paint_tsne(self, X, centroids, cluster_ids):
        # 2-step reduction: PCA -> 50d, then t-SNE -> 2d
        pca50 = PCA(n_components=50, random_state=self.config.seed)
        X50 = pca50.fit_transform(X)
        C50 = pca50.transform(centroids)

        # объединяем для единой t-SNE
        combined = np.vstack([X50, C50])
        tsne2 = TSNE(n_components=2, random_state=self.config.seed)
        Y = tsne2.fit_transform(combined)

        proj_X = Y[: X.shape[0]]
        proj_C = Y[X.shape[0] : X.shape[0] + self.config.n_ensemble_models]
        proj_sel = proj_X[self.config.selected_indices]

        os.makedirs("logs", exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.rc("font", size=20)
        plt.scatter(
            proj_X[:, 0],
            proj_X[:, 1],
            c=cluster_ids,
            cmap="tab10",
            alpha=0.4,
            label="All models",
        )
        plt.scatter(
            proj_C[:, 0],
            proj_C[:, 1],
            c="black",
            marker="X",
            s=100,
            label="Centroids",
        )
        plt.scatter(
            proj_sel[:, 0],
            proj_sel[:, 1],
            c="red",
            marker="*",
            s=150,
            label="Chosen models",
        )
        # plt.title("t-SNE проекция эмбеддингов после PCA-50")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("logs/tsne.png")
        plt.close()

    def save_models(self):
        shutil.rmtree(self.config.best_models_save_path, ignore_errors=True)
        os.makedirs(self.config.best_models_save_path, exist_ok=True)

        # Сохраняем архитектуры по одной
        for i, arch in enumerate(self.config.selected_archs, 1):
            file_path = os.path.join(
                self.config.best_models_save_path, f"model_{i:02d}.json"
            )
            with open(file_path, "w") as f:
                json.dump(arch, f, indent=4)
            print(f"Сохранена модель {i} в {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate inference")
    parser.add_argument("--hyperparameters_json", type=str, required=True)
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    params.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    config = TrainConfig(**params)

    inference = InferSurrogate(config)
    inference.initialize_models()
    inference.architecture_search()
    if config.random_choice_out_of_best:
        pass
    else:
        inference.select_central_models_by_clusters()
    inference.save_models()
