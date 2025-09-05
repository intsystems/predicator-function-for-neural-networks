import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"  
os.environ["OMP_NUM_THREADS"] = "1"       
os.environ["MKL_NUM_THREADS"] = "1" 
import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
import shutil
from pathlib import Path
import gc
import random
import re
from typing import List, Dict, Tuple, Any

from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from nni.nas.space import model_context
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist

# Custom imports
import sys

sys.path.insert(1, "../dependencies")

from dependencies.GCN import (
    GAT_ver_1,
    GAT_ver_2,
    CustomDataset,
    collate_graphs,
    extract_embeddings,
)
from dependencies.data_generator import (
    generate_arch_dicts,
    mutate_architectures,
    load_dataset_on_inference,
)
from dependencies.train_config import TrainConfig


class InferSurrogate:
    """
    Класс для поиска и выбора архитектур нейронных сетей для ансамбля
    с использованием суррогатных моделей для предсказания точности и разнообразия.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.device = torch.device(self.config.device)
        self.surrogate_dataset_path = Path(self.config.surrogate_inference_path)
        self.tmp_dir = Path(self.config.tmp_archs_path)

        self._clear_all_buffers()

    # --------------------------------------------------------------------------
    # 1. Публичные методы (API класса)
    # --------------------------------------------------------------------------

    def initialize_models(self) -> None:
        """Загружает и инициализирует суррогатные модели для точности и разнообразия."""
        print("Initializing surrogate models...")
        self.model_accuracy = GAT_ver_1(
            input_dim=self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
        ).to(self.device)
        state_dict_acc = torch.load(
            self.surrogate_dataset_path / "model_accuracy.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model_accuracy.load_state_dict(state_dict_acc)
        self.model_accuracy.eval()

        self.model_diversity = GAT_ver_2(
            input_dim=self.config.input_dim,
            output_dim=self.config.div_output_dim,
            dropout=self.config.div_dropout,
            heads=self.config.div_n_heads,
        ).to(self.device)
        state_dict_div = torch.load(
            self.surrogate_dataset_path / "model_diversity.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model_diversity.load_state_dict(state_dict_div)
        self.model_diversity.eval()
        print("Surrogate models loaded successfully.")

    def run_selection(self, strategy: str = "greedy_forward") -> None:
        """
        Основной метод, запускающий процесс отбора моделей для ансамбля.

        Args:
            strategy (str): Стратегия отбора моделей.
                - 'greedy_forward': Жадный итеративный отбор с учетом точности и разнообразия.
                - 'cluster': Отбор моделей, ближайших к центроидам кластеров.
                - 'random': Случайный отбор.
        """
        self._clear_all_buffers()
        
        # Шаг 1: Подготовка пула потенциальных моделей
        self._prepare_potential_models()
        if not self.potential_archs:
            print("Error: No potential models found after filtering. Can't select an ensemble.")
            return

        # Шаг 2: Выбор моделей в соответствии с выбранной стратегией
        print(f"\n--- Running selection with '{strategy}' strategy ---")
        n_select = self.config.n_ensemble_models
        
        if strategy == "greedy_forward":
            selected_archs = self.greedy_forward_selection(
                self.potential_archs,
                self.potential_accuracies,
                self.potential_embeddings,
                n_select,
            )
        elif strategy == "cluster":
            selected_archs, _, selected_indices = self.select_by_clusters(
                self.potential_archs,
                self.potential_embeddings,
                n_select,
                n_near_centroid=1
            )
            # Опционально: нарисовать t-SNE
            if self.config.draw_tsne:
                 self.paint_tsne(
                    self.potential_embeddings, 
                    self.potential_accuracies, # Передаем accs для полной картины
                    selected_indices
                )
        elif strategy == "random":
            selected_archs = self.select_randomly(self.potential_archs, n_select)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

        self.selected_archs = selected_archs
        print(f"\n--- Selection finished. Total models in ensemble: {len(self.selected_archs)} ---")

    def save_models(self, cleanup_tmp: bool = True) -> None:
        """Сохраняет выбранные архитектуры ансамбля в JSON-файлы."""
        if not self.selected_archs:
            print("Warning: No models were selected, nothing to save.")
            return

        base_save_path = Path(self.config.best_models_save_path)
        base_save_path.mkdir(parents=True, exist_ok=True)
        
        # Найти следующий доступный индекс для папки, чтобы не перезаписать старые результаты
        existing_dirs = base_save_path.glob("models_json_*")
        indices = [int(re.search(r"(\d+)", p.name).group(1)) for p in existing_dirs if re.search(r"(\d+)", p.name)]
        next_index = max(indices, default=0) + 1
        
        models_json_dir = base_save_path / f"models_json_{next_index}"
        models_json_dir.mkdir()
        print(f"Saving selected models to: {models_json_dir}")

        for i, arch in enumerate(self.selected_archs):
            # Имя файла на основе ID или простого индекса
            model_id = arch.get("id")
            file_name = f"model_{model_id}.json" if model_id is not None else f"generated_model_{i+1}.json"
            file_path = models_json_dir / file_name
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(arch, f, indent=4)
        
        print(f"Successfully saved {len(self.selected_archs)} model architectures.")
        
        if cleanup_tmp:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            print("Temporary directory cleaned up.")

    # --------------------------------------------------------------------------
    # 2. Стратегии отбора моделей (теперь они stateless)
    # --------------------------------------------------------------------------
    def greedy_forward_selection(
        self,
        archs: List[Dict],
        accs: np.ndarray,
        embs: np.ndarray,
        n_select: int,
    ) -> List[Dict]:
        """
        Жадный итеративный отбор (Forward Selection).
        1. Выбирает первую модель с максимальной точностью.
        2. Итеративно добавляет модели, которые максимизируют score(точность, разнообразие).
        """
        if len(archs) < n_select:
            print(f"Warning: Not enough candidates ({len(archs)}) to select {n_select}. Returning all candidates.")
            return archs
            
        candidate_accs = torch.from_numpy(accs).to(self.device).float()
        candidate_embs = torch.from_numpy(embs).to(self.device).float()
        candidate_archs = list(archs)

        # Шаг 1: Выбрать первую модель с наилучшей точностью
        best_first_idx = torch.argmax(candidate_accs).item()
        
        ensemble_archs = [candidate_archs.pop(best_first_idx)]
        ensemble_embs = [candidate_embs[best_first_idx]]
        # Для логирования можно также хранить точности
        ensemble_accs = [candidate_accs[best_first_idx]]
        
        
        print(
            f"Adding model 1/{n_select} with best accuracy = {ensemble_accs[0].item() / 100:.2f}"
        )
        
        candidate_accs = torch.cat([candidate_accs[:best_first_idx], candidate_accs[best_first_idx+1:]])
        candidate_embs = torch.cat([candidate_embs[:best_first_idx], candidate_embs[best_first_idx+1:]])
        
        while len(ensemble_archs) < n_select and len(candidate_archs) > 0:
            current_ensemble_embs = torch.stack(ensemble_embs)
            
            dists = torch.cdist(candidate_embs, current_ensemble_embs)
            mean_dists = dists.mean(dim=1)
            
            scores = self._get_score(candidate_accs, mean_dists)
            
            best_idx = torch.argmax(scores).item()
            
            print(
                f"Adding model {len(ensemble_archs) + 1}/{n_select} with "
                f"score = {scores[best_idx].item():.2f}, "
                f"accuracy = {candidate_accs[best_idx].item() / 100:.2f}, "
                f"mean_dist = {mean_dists[best_idx].item():.2f}"
            )
            
            ensemble_archs.append(candidate_archs.pop(best_idx))
            ensemble_embs.append(candidate_embs[best_idx])
            ensemble_accs.append(candidate_accs[best_idx])
            
            candidate_accs = torch.cat([candidate_accs[:best_idx], candidate_accs[best_idx+1:]])
            candidate_embs = torch.cat([candidate_embs[:best_idx], candidate_embs[best_idx+1:]])

        return ensemble_archs

    def select_by_clusters(
        self,
        archs: List[Dict],
        embs: np.ndarray,
        n_clusters: int,
        n_near_centroid: int = 1
    ) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
        """
        Выбирает модели, которые наиболее близки к центроидам кластеров.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.seed, n_init=10)
        cluster_ids = kmeans.fit_predict(embs)
        centroids = kmeans.cluster_centers_

        selected_archs, selected_embs, selected_indices = [], [], []

        for cluster_id in range(n_clusters):
            # Находим индексы всех точек в данном кластере
            idxs_in_cluster = np.where(cluster_ids == cluster_id)[0]
            if idxs_in_cluster.size == 0:
                print(f"Warning: Cluster {cluster_id} is empty.")
                continue

            cluster_embs = embs[idxs_in_cluster]
            
            # Вычисляем расстояния до центроида и находим n ближайших
            dists = np.linalg.norm(cluster_embs - centroids[cluster_id], axis=1)
            sorted_local_indices = np.argsort(dists)[:n_near_centroid]
            
            # Получаем глобальные индексы лучших моделей
            global_indices = idxs_in_cluster[sorted_local_indices]
            
            for global_idx in global_indices:
                selected_archs.append(archs[global_idx])
                selected_embs.append(embs[global_idx])
                selected_indices.append(global_idx)
                
                print(
                    f"Selected model from cluster {cluster_id} "
                    f"(dist to centroid: {dists[global_idx]:.2f})"
                )

        return selected_archs, selected_embs, selected_indices

    def select_randomly(self, archs: List[Dict], n_select: int) -> List[Dict]:
        """Выбирает n_select случайных моделей из предложенного списка."""
        if len(archs) <= n_select:
            return archs
        
        indices = random.sample(range(len(archs)), k=n_select)
        print(f"Randomly selected {len(indices)} models.")
        return [archs[i] for i in indices]

    # --------------------------------------------------------------------------
    # 3. Внутренние (служебные) методы
    # --------------------------------------------------------------------------
    
    def _prepare_potential_models(self) -> None:
        """
        Загружает/генерирует архитектуры, вычисляет их эмбеддинги и точности,
        фильтрует и сохраняет "хороших" кандидатов для дальнейшего отбора.
        """
        print("\n--- Preparing potential models pool ---")
        
        if self.config.use_pretrained_models_for_ensemble:
            # Логика загрузки готовых моделей
            load_dataset_on_inference(self.config)
            initial_archs = []
            for arch_json_path in tqdm(self.config.models_dict_path, desc="Loading pretrained archs"):
                arch = json.loads(arch_json_path.read_text(encoding="utf-8"))
                # Очистка ненужных ключей
                for key in ("test_predictions", "test_accuracy", "valid_predictions", "valid_accuracy"):
                    arch.pop(key, None)
                arch["id"] = int(re.search(r"model_(\d+)", str(arch_json_path)).group(1))
                initial_archs.append(arch)
        else:
            initial_archs = generate_arch_dicts(self.config.n_models_to_generate, use_tqdm=True)

        if not initial_archs:
            print("Error: No initial architectures were loaded or generated.")
            return

        archs, accs_np, embs_np = self._get_embeddings(initial_archs)
        
        # Шаг 3: Фильтрация по минимальной точности
        mask = (accs_np / 100)  >= self.config.min_accuracy_for_pool
        
        self.potential_archs = [arch for arch, ok in zip(archs, mask) if ok]
        self.potential_accuracies = accs_np[mask]
        self.potential_embeddings = embs_np[mask]
        
        print(f"Initial pool size: {len(initial_archs)}. "
              f"After filtering by accuracy >= {self.config.min_accuracy_for_pool}: {len(self.potential_archs)} models.")
        
        torch.cuda.empty_cache()
        gc.collect()

    def _get_embeddings(self, archs: List[Dict]) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Вычисляет предсказания точности и эмбеддинги разнообразия для списка архитектур.
        Этот метод всегда работает с чистой временной директорией.
        """
        # Очистка и создание временной директории для этого запуска
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение архитектур во временные файлы
        arch_paths = []
        for i, arch in enumerate(archs):
            path = self.tmp_dir / f"arch_{i}.json"
            path.write_text(json.dumps(arch), encoding="utf-8")
            arch_paths.append(path)
            
        dataset = CustomDataset(arch_paths, use_tqdm=True)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_inference,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_graphs,
        )

        with torch.no_grad():
            print("Extracting accuracy predictions...")
            emb_acc_np, _ = extract_embeddings(self.model_accuracy, loader, self.device, use_tqdm=True) # Должен возвращать (N, 1)
            
            print("Extracting diversity embeddings...")
            emb_div_np, _ = extract_embeddings(self.model_diversity, loader, self.device, use_tqdm=True) # Должен возвращать (N, D)

        return archs, emb_acc_np.flatten(), emb_div_np

    def _get_score(self, accuracy: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """
        Векторизованный расчет "оценки" модели.

        Args:
            accuracy (torch.Tensor): Тензор с предсказанными точностями (в процентах, 0-100).
            dist (torch.Tensor): Тензор с расстояниями.
        
        Returns:
            torch.Tensor: Тензор с итоговыми оценками.
        """
        gamma = self.config.acc_distance_gamma
        return (1 - gamma) * (accuracy / 100.0) + gamma * dist
    
    def _clear_all_buffers(self) -> None:
        """Очищает все временные списки с моделями в `self`."""
        # Буферы для пула кандидатов
        self.potential_archs: List[Dict] = []
        self.potential_embeddings: np.ndarray = np.array([])
        self.potential_accuracies: np.ndarray = np.array([])
        # Буферы для финального ансамбля
        self.selected_archs: List[Dict] = []

    def paint_tsne(self, X, centroids, cluster_ids):
        # 2-step reduction: PCA -> 50d, then t-SNE -> 2d
        pca50 = PCA(
            n_components=min(50, X.shape[0], X.shape[1]),
            random_state=self.config.seed,
        )
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
        base_save_path = Path(self.config.best_models_save_path)
        base_save_path.mkdir(parents=True, exist_ok=True)

        # Найти следующий доступный индекс
        existing_json_dirs = [
            p for p in base_save_path.glob("models_json_*") if p.is_dir()
        ]
        indices = [
            int(re.search(r"models_json_(\d+)", str(p)).group(1))
            for p in existing_json_dirs
            if re.search(r"models_json_(\d+)", str(p))
        ]
        next_index = max(indices, default=0) + 1

        # Создание новых папок с новым индексом
        models_json_dir = base_save_path / f"models_json_{next_index}"
        models_pth_dir = base_save_path / f"models_pth_{next_index}"
        models_json_dir.mkdir(parents=True, exist_ok=False)
        models_pth_dir.mkdir(parents=True, exist_ok=False)

        for i, arch in enumerate(self.config.selected_archs, 1):
            if arch.get("id") is not None:
                model_id = arch["id"]
                file_path = models_json_dir / f"model_{model_id:d}.json"
                file_path.write_text(json.dumps(arch, indent=4), encoding="utf-8")
                print(f"Saved model {i} in {file_path}")

                src = Path(
                    re.sub(
                        r"archs.*",
                        f"pth/model_{model_id:d}.pth",
                        str(self.config.prepared_dataset_path),
                    )
                )

                dst = models_pth_dir / f"model_{model_id:d}.pth"
                shutil.copy(src, dst)
            else:
                file_path = models_json_dir / f"model_{i:d}.json"
                file_path.write_text(json.dumps(arch, indent=4), encoding="utf-8")
                print(f"Saved model {i} in {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surrogate-based Ensemble Selection")
    parser.add_argument("--hyperparameters_json", type=str, required=True, 
                        help="Path to the JSON file with configuration.")
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text(encoding="utf-8"))
    config = TrainConfig(**params)


    strategy = None
    if getattr(config, 'greedy_choice_out_of_best', False):
        strategy = 'greedy_forward'
    elif getattr(config, 'random_choice_out_of_best', False):
        strategy = 'random'
    else:
        strategy = 'cluster'
    
    print(f"Chosen selection strategy: '{strategy}'")
    inference = InferSurrogate(config)
    
    inference.initialize_models()

    try:
        inference.run_selection(strategy=strategy) # <<-- Передаем сюда нашу переменную
    except ValueError as e:
        print(f"Error during selection: {e}")
        exit(1)

    inference.save_models()

    print("\nProcess finished successfully!")

