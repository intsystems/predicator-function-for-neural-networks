import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import argparse
import shutil
from pathlib import Path
import gc
import random
import re

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferSurrogate:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(self.config.device)
        self.dataset_path = Path(self.config.surrogate_inference_path)

    def initialize_models(self):
        self.config.model_accuracy = GAT_ver_1(
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

        self.config.model_diversity = GAT_ver_2(
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

    def _get_score(self, accuracy, dist):
        return (
            1 - self.config.acc_distance_gamma
        ) * accuracy / 100 + self.config.acc_distance_gamma * dist

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

        if self.config.use_pretrained_models_for_ensemble:
            load_dataset_on_inference(self.config)
            arch_dicts = []
            for arch_json in tqdm(self.config.models_dict_path, desc="Deleting keys"):
                arch = json.loads(arch_json.read_text(encoding="utf-8"))
                for key in (
                    "test_predictions",
                    "test_accuracy",
                    "valid_predictions",
                    "valid_accuracy",
                ):
                    arch.pop(key, None)
                arch["id"] = int(re.search(r"model_(\d+)", str(arch_json)).group(1))
                arch_dicts.append(arch)
        else:
            arch_dicts = generate_arch_dicts(
                self.config.n_models_to_generate, use_tqdm=True
            )

        arch_paths, emb_acc_np, emb_div_np = self._get_embeddings(tmp_dir, arch_dicts)

        mask = emb_acc_np >= self.config.min_accuracy_for_pool
        valid_paths = [p for p, ok in zip(arch_paths, mask) if ok]
        valid_div_embs = emb_div_np[mask].astype(np.float16)
        valid_accs = emb_acc_np[mask]

        for path, emb, acc in zip(valid_paths, valid_div_embs, valid_accs):
            arch = json.loads(path.read_text(encoding="utf-8"))
            self.config.potential_archs.append(arch)
            self.config.potential_embeddings.append(emb)
            self.config.potential_accuracies.append(acc)

        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def select_models_only_score(self, n_dist_calc=10):
        models_with_scores = []

        for idx, (arch, emb, acc) in enumerate(
            tqdm(
                zip(
                    self.config.potential_archs,
                    self.config.potential_embeddings,
                    self.config.potential_accuracies,
                ),
                desc="Calculating scores",
            )
        ):
            neighbors = np.array(
                random.choices(
                    self.config.potential_embeddings,
                    k=n_dist_calc,
                )
            )
            dist = np.mean(np.linalg.norm(neighbors - emb, axis=1))
            score = self._get_score(acc, dist)
            models_with_scores.append((arch, emb, acc, idx, score))

        models_with_scores.sort(key=lambda x: x[4], reverse=True)

        for i in range(self.config.n_ensemble_models):
            self.config.selected_archs.append(models_with_scores[i][0])
            self.config.selected_embs.append(models_with_scores[i][1])
            self.config.selected_accs.append(models_with_scores[i][2])
            self.config.selected_indices.append(models_with_scores[i][3])
            print(
                f"Adding model {i+1}/{self.config.n_ensemble_models} with score = {models_with_scores[i][4]:.2f}, accuracy = {models_with_scores[i][2]:.2f}"
            )

    def select_central_models_by_clusters(self, n_near_centroid=1, draw_tsne=True):

        X = np.array(self.config.potential_embeddings, dtype=np.float32)
        accs = np.array(self.config.potential_accuracies, dtype=np.float32)

        kmeans = KMeans(
            n_clusters=self.config.n_ensemble_models,
            random_state=self.config.seed,
            n_init=10,
        )
        cluster_ids = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        self.config.selected_archs = []
        self.config.selected_embs = []
        self.config.selected_accs = []
        self.config.selected_indices = []

        for cluster_id in range(self.config.n_ensemble_models):
            idxs = np.where(cluster_ids == cluster_id)[0]
            if idxs.size == 0:
                continue

            cluster_X = X[idxs]
            centroid = centroids[cluster_id]

            dists = np.linalg.norm(cluster_X - centroid, axis=1)
            sorted_local = np.argsort(dists)[:n_near_centroid]

            for local_best in sorted_local:
                best_global = idxs[local_best]

                self.config.selected_archs.append(
                    self.config.potential_archs[best_global]
                )
                self.config.selected_embs.append(X[best_global])
                self.config.selected_accs.append(accs[best_global])
                self.config.selected_indices.append(best_global)

        if draw_tsne:
            self.paint_tsne(X, centroids, cluster_ids)

    def random_choice_out_of_best(self):
        X = np.array(self.config.potential_embeddings, dtype=np.float32)
        accs = np.array(self.config.potential_accuracies, dtype=np.float32)

        for i in range(self.config.n_ensemble_models):
            chosen = np.random.randint(X.shape[0])
            self.config.selected_archs.append(self.config.potential_archs[chosen])
            self.config.selected_embs.append(X[chosen])
            self.config.selected_accs.append(accs[chosen])
            self.config.selected_indices.append(chosen)

    def _clear_buffers(self, potential=True, seleceted=True):
        if potential:
            self.config.potential_archs = []
            self.config.potential_embeddings = []
            self.config.potential_accuracies = []
        if seleceted:
            self.config.selected_archs = []
            self.config.selected_embs = []
            self.config.selected_accs = []
            self.config.selected_indices = []

    def _get_embeddings(self, path_dir, archs):
        shutil.rmtree(path_dir, ignore_errors=True)
        path_dir.mkdir(parents=True, exist_ok=True)

        arch_paths = []
        for i, arch in enumerate(archs):
            path = path_dir / f"arch_{len(os.listdir(path_dir)) + i}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(arch, f)
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
            emb_acc_np, _ = extract_embeddings(
                self.config.model_accuracy, loader, device, use_tqdm=True
            )
            emb_div_np, _ = extract_embeddings(
                self.config.model_diversity, loader, device, use_tqdm=True
            )

        return arch_paths, emb_acc_np, emb_div_np

    def _select_top_diverse_models(self, acc_embs, div_embs):
        n_models = len(acc_embs)
        n_keep = min(self.config.n_ensemble_models * 20, n_models)

        acc_embs = np.array(acc_embs)
        div_embs = np.array(div_embs)

        first_idx = int(np.argmax(acc_embs))
        selected_idxs = [first_idx]
        remaining_idxs = set(range(n_models)) - {first_idx}

        scores = [0.0]
        accs = [acc_embs[first_idx]]
        dists = [0.0]

        while len(selected_idxs) < n_keep:
            best_idx, best_score = None, -np.inf

            for idx in remaining_idxs:
                distances = cdist(div_embs[[idx]], div_embs[selected_idxs], metric="euclidean")[0]
                mean_dist = distances.mean()
                score = self._get_score(acc_embs[idx], mean_dist)

                if score > best_score:
                    best_score, best_idx = score, idx
                    best_distance = mean_dist

            selected_idxs.append(best_idx)
            scores.append(best_score)
            accs.append(acc_embs[best_idx])
            dists.append(best_distance)
            remaining_idxs.remove(best_idx)

        if len(selected_idxs) > 1:
            first_distances = cdist(div_embs[[first_idx]], div_embs[selected_idxs[1:]], metric="euclidean")[0]
            scores[0] = self._get_score(acc_embs[first_idx], first_distances.mean())
            dists[0] = first_distances.mean()

        self.config.selected_archs   = [self.config.selected_archs[i]   for i in selected_idxs]
        self.config.selected_embs    = [self.config.selected_embs[i]    for i in selected_idxs]
        self.config.selected_accs    = [self.config.selected_accs[i]    for i in selected_idxs]
        self.config.selected_indices = [self.config.selected_indices[i] for i in selected_idxs]

        return scores, accs, dists

    def greedy_choice_out_of_best(self):
        tmp_dir = Path(self.config.tmp_archs_path)

        ensemble_list = []
        tmp_archs = generate_arch_dicts(self.config.n_models_in_pool)
        while len(ensemble_list) < self.config.n_ensemble_models:
            self._clear_buffers()

            arch_paths, emb_acc_np, emb_div_np = self._get_embeddings(
                tmp_dir, tmp_archs
            )

            mask = emb_acc_np >= self.config.min_accuracy_for_pool
            valid_paths = [p for p, ok in zip(arch_paths, mask) if ok]
            emb_div_np = emb_div_np[mask]
            emb_acc_np = emb_acc_np[mask]

            for path, emb, acc in zip(valid_paths, emb_div_np, emb_acc_np):
                arch = json.loads(path.read_text(encoding="utf-8"))
                self.config.potential_archs.append(arch)
                self.config.potential_embeddings.append(emb)
                self.config.potential_accuracies.append(acc)

            if len(ensemble_list) == 0:
                self.select_central_models_by_clusters(draw_tsne=False)
                best_model = self.config.selected_archs[0]
                ensemble_list.append(best_model)
            else:
                self.select_central_models_by_clusters(
                    n_near_centroid=10, draw_tsne=False
                )

                _, emb_acc_np, emb_div_np = self._get_embeddings(
                    tmp_dir, self.config.selected_archs
                )

                if len(emb_acc_np) == 0:  # if no models are above min_accuracy_for_pool
                    tmp_archs = mutate_architectures(
                        tmp_archs,
                        n_mutations=4,
                        n_out_models=self.config.n_models_to_generate,
                    )
                    tmp_archs.extend(ensemble_list)
                    continue

                scores, accs, dists = self._select_top_diverse_models(emb_acc_np, emb_div_np)

                for idx, score in enumerate(scores):
                    if score >= self.config.min_acc_and_div_to_ensemble:
                        print(
                            f"Adding model {len(ensemble_list) + 1}/{self.config.n_ensemble_models} with score = {score:.2f}, "
                            f"accuracy = {accs[idx] / 100:.2f}, "
                            f"distance = {dists[idx]:.2f}"
                        )
                        ensemble_list.append(self.config.selected_archs[idx])
                        if len(ensemble_list) >= self.config.n_ensemble_models:
                            break

            tmp_archs = mutate_architectures(
                ensemble_list,
                n_mutations=4,
                n_out_models=self.config.n_models_to_generate,
            )
            tmp_archs.extend(ensemble_list)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        self.config.selected_archs = ensemble_list

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
    parser = argparse.ArgumentParser(description="Surrogate inference")
    parser.add_argument("--hyperparameters_json", type=str, required=True)
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    params.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    config = TrainConfig(**params)

    inference = InferSurrogate(config)
    inference.initialize_models()

    if config.random_choice_out_of_best:
        inference.architecture_search()
        inference.random_choice_out_of_best()
    elif config.greedy_choice_out_of_best:
        inference.greedy_choice_out_of_best()
    elif config.no_clusters_choice:
        inference.architecture_search()
        inference.select_models_only_score()
    else:
        inference.architecture_search()
        inference.select_central_models_by_clusters()
    inference.save_models()
