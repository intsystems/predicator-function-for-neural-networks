import argparse
import copy
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Добавляем путь к корню проекта для импорта модулей
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from dependencies.GCN import (
    GAT_ver_2,
    CustomDataset,
    collate_graphs,
    collate_triplets,
    extract_embeddings,
    train_model_diversity,
)
from dependencies.data_generator import load_dataset
from dependencies.train_config import TrainConfig
from train_surrogate import SurrogateTrainer

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/surrogate_hp_CIFAR100.json"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "ablation_study/diversity_surrogate/results"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "ablation_study/diversity_surrogate/correlation_analysis"
DEFAULT_MATRIX_CACHE_PATH = PROJECT_ROOT / "ablation_study/diversity_surrogate/diversity_matrix_cache.npz"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Diversity surrogate ablation study")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
    )
    parser.add_argument(
        "--model_counts",
        type=int,
        nargs="+",
        default=[0, 2000, 1500, 1000, 500, 250],
        help="Explicit cumulative model counts to train/evaluate",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Load existing checkpoints if present",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Directory for diversity checkpoints and JSON results",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for plots",
    )
    parser.add_argument(
        "--matrix_cache_path",
        type=str,
        default=str(DEFAULT_MATRIX_CACHE_PATH),
        help="Path to cached diversity matrices (.npz)",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=1000,
        help="Number of test-test pairs for correlation evaluation",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of repeated runs for each training set size",
    )
    return parser.parse_args()


class IndexedSubsetDataset(Dataset):
    def __init__(
        self,
        base_paths: List[Path],
        indices: List[int],
        accuracies: np.ndarray,
        preload: bool = False,
    ):
        self.base_paths = base_paths
        self.indices = list(indices)
        self.accuracies = accuracies
        self._cache: Dict[int, object] = {}

        if preload:
            for idx in tqdm(range(len(self.indices)), desc="Preloading graphs", leave=False):
                self._cache[idx] = self._build_data(idx)

    def __len__(self):
        return len(self.indices)

    def _build_data(self, idx):
        original_idx = self.indices[idx]
        path = self.base_paths[original_idx]
        with path.open("r", encoding="utf-8") as f:
            model_dict = json.load(f)

        parsed_model_id = None
        match = re.search(r"model_(\d+)", str(path))
        if match:
            parsed_model_id = int(match.group(1))

        from dependencies.Graph import Graph
        from torch_geometric.utils import dense_to_sparse
        from torch_geometric.data import Data

        graph = Graph(model_dict, index=original_idx)
        adj, _, features = graph.get_adjacency_matrix()
        adj, features = CustomDataset.preprocess(adj, features)
        edge_index, _ = dense_to_sparse(adj)

        data = Data(x=features, edge_index=edge_index)
        data.index = original_idx
        data.sample_id = torch.tensor([original_idx], dtype=torch.long)
        data.dataset_position = idx
        data.path_position = original_idx
        data.parsed_model_id = -1 if parsed_model_id is None else parsed_model_id
        data.source_path = str(path)
        data.y = torch.tensor(self.accuracies[original_idx], dtype=torch.float)
        return data

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._build_data(idx)
        return self._cache[idx].clone()


class IndexedTripletDataset(Dataset):
    def __init__(self, base_dataset: IndexedSubsetDataset, diversity_matrix: np.ndarray):
        self.base = base_dataset
        self.div = diversity_matrix
        self.N = len(self.base)
        self.orig2int = {self.base.indices[i]: i for i in range(self.N)}
        self.valid_anchor_positions = self._build_valid_anchor_positions()

    def _build_valid_anchor_positions(self) -> List[int]:
        valid_positions = []
        for pos, anchor_orig in enumerate(self.base.indices):
            row = self.div[anchor_orig]
            pos_orig = np.where((row == 1) & (np.arange(len(row)) != anchor_orig))[0]
            neg_orig = np.where(row == -1)[0]
            pos_orig = [i for i in pos_orig if i in self.orig2int]
            neg_orig = [i for i in neg_orig if i in self.orig2int]
            if pos_orig and neg_orig:
                valid_positions.append(pos)
        return valid_positions

    def __len__(self):
        return len(self.valid_anchor_positions)

    def __getitem__(self, idx):
        anchor_pos = self.valid_anchor_positions[idx]
        anchor = self.base[anchor_pos]
        anchor_orig = self.base.indices[anchor_pos]

        row = self.div[anchor_orig]
        pos_orig = np.where((row == 1) & (np.arange(len(row)) != anchor_orig))[0]
        neg_orig = np.where(row == -1)[0]
        pos_orig = [i for i in pos_orig if i in self.orig2int]
        neg_orig = [i for i in neg_orig if i in self.orig2int]

        pos_o = int(np.random.choice(pos_orig))
        neg_o = int(np.random.choice(neg_orig))

        positive = self.base[self.orig2int[pos_o]]
        negative = self.base[self.orig2int[neg_o]]
        idx_triplet = torch.tensor([anchor_orig, pos_o, neg_o], dtype=torch.long)
        return anchor, positive, negative, idx_triplet


class DiversityAblationTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dataset_path = Path(config.dataset_path)
        self.all_model_paths: List[Path] = []
        self.all_accuracies: Optional[np.ndarray] = None
        self.initial_archs: List[Dict] = []
        self.diversity_matrix: Optional[np.ndarray] = None
        self.discrete_diversity_matrix: Optional[np.ndarray] = None
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []
        self.test_test_pairs: List[Tuple[int, int]] = []

    def load_all_models(self) -> None:
        load_dataset(self.config)
        self.all_model_paths = sorted(self.dataset_path.rglob("*.json"))
        self.config.models_dict_path = list(self.all_model_paths)
        self.config.div_models_dict_path = list(self.all_model_paths)
        self.config.acc_models_dict_path = list(self.all_model_paths)

        print(f"Total models available: {len(self.all_model_paths)}")
        self.initial_archs = []
        accuracies = []

        for arch_json_path in tqdm(self.all_model_paths, desc="Loading pretrained archs"):
            arch = json.loads(arch_json_path.read_text(encoding="utf-8"))
            accuracies.append(arch.get("valid_accuracy", arch.get("test_accuracy", 0.0)) * 100)
            for key in (
                "test_predictions",
                "test_accuracy",
                "valid_predictions",
                "valid_accuracy",
            ):
                arch.pop(key, None)
            arch["id"] = int(re.search(r"model_(\d+)", str(arch_json_path)).group(1))
            self.initial_archs.append(arch)

        self.all_accuracies = np.array(accuracies, dtype=float)

    def prepare_common_data(
        self,
        n_pairs: int = 1000,
        random_seed: int = 42,
        matrix_cache_path: Optional[Path] = None,
    ) -> None:
        cache_path = Path(matrix_cache_path) if matrix_cache_path else None

        if cache_path and cache_path.exists():
            print(f"\nLoading diversity matrix from cache: {cache_path}")
            cached = np.load(cache_path)
            self.diversity_matrix = cached["diversity_matrix"]
            self.discrete_diversity_matrix = cached["discrete_diversity_matrix"]
        else:
            print("\nComputing diversity matrix once...")
            trainer = SurrogateTrainer(self.config)
            trainer.get_diversity_matrix()
            trainer.create_discrete_diversity_matrix()
            self.diversity_matrix = trainer.config.diversity_matrix
            self.discrete_diversity_matrix = trainer.config.discrete_diversity_matrix

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    diversity_matrix=self.diversity_matrix,
                    discrete_diversity_matrix=self.discrete_diversity_matrix,
                )
                print(f"Saved diversity matrix cache to: {cache_path}")

        print(f"Diversity matrix shape: {self.diversity_matrix.shape}")

        indices = np.arange(len(self.all_model_paths))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.seed(None)

        train_size = int(0.8 * len(indices))
        self.train_indices = list(indices[:train_size])
        self.test_indices = list(indices[train_size:])
        print(f"Train pool size: {len(self.train_indices)}")
        print(f"Test pool size: {len(self.test_indices)}")

        max_test_pairs = len(self.test_indices) * (len(self.test_indices) - 1) // 2
        actual_n_pairs = min(n_pairs, max_test_pairs)
        self.test_test_pairs = get_random_pairs(self.test_indices, self.test_indices, actual_n_pairs)
        print(f"Created {len(self.test_test_pairs)} test-test pairs")

    def create_subset_dataset(
        self,
        indices: List[int],
        preload: bool = False,
    ) -> IndexedSubsetDataset:
        return IndexedSubsetDataset(
            self.all_model_paths,
            indices,
            self.all_accuracies,
            preload=preload,
        )

    def create_triplet_dataloaders(self, indices: List[int]):
        subset_dataset = self.create_subset_dataset(indices, preload=True)
        subset_triplet_dataset = IndexedTripletDataset(subset_dataset, self.discrete_diversity_matrix)

        if len(subset_triplet_dataset) == 0:
            raise ValueError(
                f"No valid triplets can be formed for subset of size {len(indices)}"
            )

        valid_triplet_count = max(1, int(round((1 - self.config.train_size) * len(subset_triplet_dataset))))
        train_triplet_count = len(subset_triplet_dataset) - valid_triplet_count
        if train_triplet_count <= 0:
            train_triplet_count = len(subset_triplet_dataset) - 1
            valid_triplet_count = 1

        train_triplet_ds, valid_triplet_ds = torch.utils.data.random_split(
            subset_triplet_dataset,
            [train_triplet_count, valid_triplet_count],
        )

        safe_num_workers = min(self.config.num_workers, 8)
        loader_kwargs = {
            "num_workers": safe_num_workers,
            "pin_memory": False,
            "collate_fn": collate_triplets,
        }
        if safe_num_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = 2

        print(
            "[triplet_dataloader] "
            f"subset_size={len(indices)}, "
            f"triplets={len(subset_triplet_dataset)}, "
            f"train_triplets={train_triplet_count}, "
            f"valid_triplets={valid_triplet_count}, "
            f"num_workers={loader_kwargs['num_workers']}, "
            f"pin_memory={loader_kwargs['pin_memory']}, "
            f"persistent_workers={loader_kwargs.get('persistent_workers', False)}, "
            f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
        )

        train_loader = DataLoader(
            train_triplet_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_triplet_ds,
            batch_size=1024,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, valid_loader

    def build_diversity_model(self):
        return GAT_ver_2(
            self.config.input_dim,
            self.config.div_output_dim,
            dropout=self.config.div_dropout,
            heads=self.config.div_n_heads,
            output_activation="l2",
            pre_norm=True,
        ).to(self.device)

    def train_diversity_model(self, indices: List[int]):
        model = self.build_diversity_model()
        print(
            "[train_diversity_model] "
            f"indices_count={len(indices)}, "
            f"device={self.device}, "
            f"batch_size={self.config.batch_size}, "
            f"num_workers={self.config.num_workers}"
        )
        train_loader, valid_loader = self.create_triplet_dataloaders(indices)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.div_lr_start)
        train_model_diversity(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            criterion=lambda a, p, n: self._triplet_loss(a, p, n),
            num_epochs=self.config.div_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.div_lr_end,
        )
        model.eval()
        return model

    def load_diversity_model(self, checkpoint_path: Path):
        model = self.build_diversity_model()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def save_diversity_model(model, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model_diversity.pth")

    def compute_embeddings_for_model(self, model_diversity):
        dataset = self.create_subset_dataset(
            list(range(len(self.all_model_paths))),
            preload=True,
        )
        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": False,
            "collate_fn": collate_graphs,
        }

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_inference,
            shuffle=False,
            **loader_kwargs,
        )

        embs_np, indices = extract_embeddings(model_diversity, loader, self.device, use_tqdm=True)
        ordered_embs = np.zeros((len(self.all_model_paths), embs_np.shape[-1]), dtype=embs_np.dtype)

        if len(indices) != len(embs_np):
            raise ValueError(
                "Embedding extraction length mismatch: "
                f"len(indices)={len(indices)}, len(embs)={len(embs_np)}"
            )

        invalid_mask = (indices < 0) | (indices >= len(self.all_model_paths))
        if np.any(invalid_mask):
            invalid_indices = indices[invalid_mask]
            raise IndexError(
                "Embedding sample ids exceed available model paths. "
                f"sample={invalid_indices[:10].tolist()}, "
                f"allowed_range=[0, {len(self.all_model_paths) - 1}]"
            )

        ordered_embs[indices] = embs_np
        return ordered_embs

    def evaluate_embeddings(self, embs_np):
        similarities = []
        distances = []
        for i, j in self.test_test_pairs:
            similarities.append(self.diversity_matrix[i, j])
            distances.append(np.linalg.norm(embs_np[i] - embs_np[j]))

        similarities = np.array(similarities)
        distances = np.array(distances)
        spearman_corr, spearman_p = spearmanr(distances, similarities)
        kendall_corr, kendall_p = kendalltau(distances, similarities)
        recall_metrics = compute_recall_at_k_for_model(
            embs_np,
            self.diversity_matrix,
            self.test_indices,
            k_values=(10, 50, 100),
        )
        return {
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "kendall_corr": float(kendall_corr),
            "kendall_p": float(kendall_p),
            "recall_at_10": float(recall_metrics["recall_at_10"]),
            "recall_at_50": float(recall_metrics["recall_at_50"]),
            "recall_at_100": float(recall_metrics["recall_at_100"]),
        }

    def _triplet_loss(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d_ap = (a - p).pow(2).sum(-1)
        d_an = (a - n).pow(2).sum(-1)
        return F.relu(d_ap - d_an + self.config.margin).mean()




def get_random_pairs(indices1, indices2, n_pairs):
    """Создает список уникальных пар индексов (i, j), где i != j."""
    pairs = set()
    while len(pairs) < n_pairs:
        i = np.random.choice(indices1)
        j = np.random.choice(indices2)
        if i != j:
            pairs.add(tuple(sorted((i, j))))
    return list(pairs)


def compute_recall_at_k_for_model(embs_np, diversity_matrix, test_indices, k_values=(10, 50, 100)):
    test_indices = np.array(test_indices)
    if len(test_indices) <= 1:
        return {f"recall_at_{k}": np.nan for k in k_values}

    max_neighbors = len(test_indices) - 1
    effective_k_values = [min(k, max_neighbors) for k in k_values]
    recall_per_k = {k: [] for k in k_values}

    for query_idx in test_indices:
        candidate_indices = test_indices[test_indices != query_idx]
        true_similarities = diversity_matrix[query_idx, candidate_indices]
        true_order = candidate_indices[np.argsort(-true_similarities)]
        latent_distances = np.linalg.norm(embs_np[candidate_indices] - embs_np[query_idx], axis=1)
        latent_order = candidate_indices[np.argsort(latent_distances)]

        for k, effective_k in zip(k_values, effective_k_values):
            true_topk = set(true_order[:effective_k])
            latent_topk = set(latent_order[:effective_k])
            recall_value = len(true_topk & latent_topk) / effective_k
            recall_per_k[k].append(recall_value)

    result = {}
    for k in k_values:
        per_query = np.array(recall_per_k[k], dtype=float)
        result[f"recall_at_{k}"] = float(np.mean(per_query)) if len(per_query) > 0 else np.nan
    return result


def compute_mean_and_variance(data):
    if len(data) < 1:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    std = np.std(data)
    var = np.var(data)
    return mean, std, var


def aggregate_results_by_train_size(results: List[Dict]) -> List[Dict]:
    grouped_results: Dict[int, List[Dict]] = defaultdict(list)
    for result in results:
        grouped_results[int(result["n_models"])].append(result)

    aggregated = []
    metric_names = [
        "spearman_corr",
        "kendall_corr",
        "recall_at_10",
        "recall_at_50",
        "recall_at_100",
    ]

    for train_size in sorted(grouped_results):
        runs = grouped_results[train_size]
        metrics_by_name = {
            metric_name: np.array([run["metrics"][metric_name] for run in runs], dtype=float)
            for metric_name in metric_names
        }
        aggregated.append(
            {
                "train_size": train_size,
                "num_runs": len(runs),
                "runs": runs,
                "summary": {
                    metric_name: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "var": float(np.var(values)),
                    }
                    for metric_name, values in metrics_by_name.items()
                },
            }
        )
    return aggregated



def plot_recall_vs_samples(aggregated_results, output_dir: Path):
    valid_results = [r for r in aggregated_results if r.get("summary")]
    if not valid_results:
        print("No valid recall results to plot.")
        return []

    train_sizes = np.array([r["train_size"] for r in valid_results])
    recall10_means = np.array([r["summary"]["recall_at_10"]["mean"] for r in valid_results])
    recall50_means = np.array([r["summary"]["recall_at_50"]["mean"] for r in valid_results])
    recall100_means = np.array([r["summary"]["recall_at_100"]["mean"] for r in valid_results])

    recall10_stds = np.array([r["summary"]["recall_at_10"]["std"] for r in valid_results])
    recall50_stds = np.array([r["summary"]["recall_at_50"]["std"] for r in valid_results])
    recall100_stds = np.array([r["summary"]["recall_at_100"]["std"] for r in valid_results])

    output_dir.mkdir(parents=True, exist_ok=True)
    recall_specs = [
        (10, recall10_means, recall10_stds, "#3C8D2F", "o", output_dir / "recall_at_10_vs_samples.pdf"),
        (50, recall50_means, recall50_stds, "#E67E22", "s", output_dir / "recall_at_50_vs_samples.pdf"),
        (100, recall100_means, recall100_stds, "#8E44AD", "^", output_dir / "recall_at_100_vs_samples.pdf"),
    ]

    figures = []
    margin = (train_sizes[-1] - train_sizes[0]) * 0.1 if len(train_sizes) > 1 else 1
    for k, means, stds, color, marker, output_path in recall_specs:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(
            train_sizes,
            means,
            f"{marker}-",
            color=color,
            linewidth=1.5,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=color,
        )
        ax.fill_between(
            train_sizes,
            np.clip(means - stds, 0.0, 1.0),
            np.clip(means + stds, 0.0, 1.0),
            alpha=0.2,
            color=color,
            linewidth=0,
        )
        ax.set_xlabel("Training set size", fontsize=12)
        ax.set_ylabel(f"Recall@{k}", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xticks(train_sizes)
        ax.set_xticklabels([str(x) for x in train_sizes])
        ax.set_xlim(train_sizes[0] - margin, train_sizes[-1] + margin)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig)
    return figures


def plot_correlation_vs_samples(aggregated_results, output_dir: Path):
    valid_results = [r for r in aggregated_results if r.get("summary")]
    if not valid_results:
        print("No valid correlation results to plot.")
        return None, None

    train_sizes = np.array([r["train_size"] for r in valid_results])
    spearman_means = np.array([r["summary"]["spearman_corr"]["mean"] for r in valid_results])
    kendall_means = np.array([r["summary"]["kendall_corr"]["mean"] for r in valid_results])
    spearman_stds = np.array([r["summary"]["spearman_corr"]["std"] for r in valid_results])
    kendall_stds = np.array([r["summary"]["kendall_corr"]["std"] for r in valid_results])

    output_dir.mkdir(parents=True, exist_ok=True)
    margin = (train_sizes[-1] - train_sizes[0]) * 0.1 if len(train_sizes) > 1 else 1

    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))
    ax1.plot(
        train_sizes,
        (-1) * spearman_means,
        "o-",
        color="#2E86AB",
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor="#2E86AB",
    )
    ax1.fill_between(
        train_sizes,
        (-1) * spearman_means - spearman_stds,
        (-1) * spearman_means + spearman_stds,
        alpha=0.2,
        color="#2E86AB",
        linewidth=0,
    )
    ax1.set_xlabel("Training set size", fontsize=12)
    ax1.set_ylabel(r"$\rho$", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_xticks(train_sizes)
    ax1.set_xticklabels([str(x) for x in train_sizes])
    ax1.set_xlim(train_sizes[0] - margin, train_sizes[-1] + margin)
    ax1.set_ylim(-1, 0)
    plt.tight_layout()
    fig1.savefig(output_dir / "spearman_vs_samples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
    ax2.plot(
        train_sizes,
        kendall_means,
        "s-",
        color="#A23B72",
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor="#A23B72",
    )
    ax2.fill_between(
        train_sizes,
        kendall_means - kendall_stds,
        kendall_means + kendall_stds,
        alpha=0.2,
        color="#A23B72",
        linewidth=0,
    )
    ax2.set_xlabel("Training set size", fontsize=12)
    ax2.set_ylabel(r"$\tau$", fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.set_xticks(train_sizes)
    ax2.set_xticklabels([str(x) for x in train_sizes])
    ax2.set_xlim(train_sizes[0] - margin, train_sizes[-1] + margin)
    ax2.set_ylim(-1, 0)
    plt.tight_layout()
    fig2.savefig(output_dir / "kendall_vs_samples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    summary = []
    for result in valid_results:
        metrics = result["summary"]
        summary.append(
            {
                "train_size": int(result["train_size"]),
                "num_runs": int(result["num_runs"]),
                "spearman_mean": float(metrics["spearman_corr"]["mean"]),
                "spearman_std": float(metrics["spearman_corr"]["std"]),
                "spearman_var": float(metrics["spearman_corr"]["var"]),
                "kendall_mean": float(metrics["kendall_corr"]["mean"]),
                "kendall_std": float(metrics["kendall_corr"]["std"]),
                "kendall_var": float(metrics["kendall_corr"]["var"]),
                "recall_at_10_mean": float(metrics["recall_at_10"]["mean"]),
                "recall_at_10_std": float(metrics["recall_at_10"]["std"]),
                "recall_at_10_var": float(metrics["recall_at_10"]["var"]),
                "recall_at_50_mean": float(metrics["recall_at_50"]["mean"]),
                "recall_at_50_std": float(metrics["recall_at_50"]["std"]),
                "recall_at_50_var": float(metrics["recall_at_50"]["var"]),
                "recall_at_100_mean": float(metrics["recall_at_100"]["mean"]),
                "recall_at_100_std": float(metrics["recall_at_100"]["std"]),
                "recall_at_100_var": float(metrics["recall_at_100"]["var"]),
            }
        )

    with (output_dir / "correlation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return fig1, fig2


def build_model_counts(requested_counts: List[int], total_models: int) -> List[int]:
    valid_counts = []
    for count in requested_counts:
        if count < 0:
            continue
        if count > total_models:
            raise ValueError(
                f"Requested model count {count} exceeds available train pool size {total_models}"
            )
        valid_counts.append(int(count))
    return sorted(set(valid_counts))


def save_results(output_base: Path, results: List[Dict]) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    with (output_base / "ablation_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_summary(aggregated_results: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("DIVERSITY ABLATION SUMMARY")
    print("=" * 60)
    if not aggregated_results:
        print("No results collected.")
        return

    for result in aggregated_results:
        metrics = result.get("summary")
        if not metrics:
            print(f"  n_models={result['train_size']:4d}: no metrics")
            continue
        print(
            f"  n_models={result['train_size']:4d} (runs={result['num_runs']}): "
            f"Spearman={metrics['spearman_corr']['mean']:.4f}±{metrics['spearman_corr']['std']:.4f}, "
            f"Kendall={metrics['kendall_corr']['mean']:.4f}±{metrics['kendall_corr']['std']:.4f}, "
            f"Recall@10={metrics['recall_at_10']['mean']:.4f}±{metrics['recall_at_10']['std']:.4f}, "
            f"Recall@50={metrics['recall_at_50']['mean']:.4f}±{metrics['recall_at_50']['std']:.4f}, "
            f"Recall@100={metrics['recall_at_100']['mean']:.4f}±{metrics['recall_at_100']['std']:.4f}"
        )


def main():
    args = parse_arguments()
    if args.num_repeats <= 0:
        raise ValueError("--num_repeats must be a positive integer")

    params = json.loads(Path(args.hyperparameters_json).read_text())
    base_seed = int(params["seed"])

    output_base = Path(args.output_base)
    plots_dir = Path(args.plots_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    model_counts: Optional[List[int]] = None

    for repeat_idx in range(args.num_repeats):
        repeat_seed = base_seed + repeat_idx
        repeat_params = dict(params)
        repeat_params["seed"] = repeat_seed
        config = TrainConfig(**repeat_params)

        trainer = DiversityAblationTrainer(config)
        print("Loading all models...")
        trainer.load_all_models()
        trainer.prepare_common_data(
            n_pairs=args.n_pairs,
            random_seed=repeat_seed,
            matrix_cache_path=Path(args.matrix_cache_path),
        )

        if model_counts is None:
            model_counts = build_model_counts(args.model_counts, len(trainer.train_indices))
            if not model_counts:
                print("No model counts to process.")
                return

        cumulative_indices: List[int] = []

        print(f"\n{'#' * 60}")
        print(f"Starting repeat {repeat_idx + 1}/{args.num_repeats} with seed={repeat_seed}")
        print(f"{'#' * 60}")

        for iteration_idx, n_models in enumerate(model_counts):
            save_dir = output_base / f"repeat_{repeat_idx + 1}" / f"train_size_{n_models}"
            checkpoint_path = save_dir / "model_diversity.pth"

            print(f"\n{'=' * 60}")
            print(f"Processing {n_models} models (cumulative), repeat {repeat_idx + 1}/{args.num_repeats}")
            print(f"{'=' * 60}")

            if iteration_idx == 0:
                if n_models == 0:
                    cumulative_indices = []
                else:
                    np.random.seed(repeat_seed)
                    cumulative_indices = list(np.random.choice(trainer.train_indices, size=n_models, replace=False))
                    np.random.seed(None)
            else:
                candidates = [i for i in trainer.train_indices if i not in cumulative_indices]
                n_to_add = n_models - len(cumulative_indices)
                if n_to_add > 0:
                    np.random.seed(repeat_seed + iteration_idx)
                    new_indices = list(np.random.choice(candidates, size=n_to_add, replace=False))
                    np.random.seed(None)
                    cumulative_indices = cumulative_indices + new_indices

            if args.skip_existing and checkpoint_path.exists():
                print(f"Loading diversity surrogate from {checkpoint_path}...")
                model_diversity = trainer.load_diversity_model(checkpoint_path)
                loaded_from_checkpoint = True
            else:
                if n_models == 0:
                    print("Initializing diversity surrogate with random weights (no training)...")
                    model_diversity = trainer.build_diversity_model()
                else:
                    print(f"Training diversity surrogate on {len(cumulative_indices)} models...")
                    model_diversity = trainer.train_diversity_model(cumulative_indices)
                trainer.save_diversity_model(model_diversity, save_dir)
                loaded_from_checkpoint = False

            print("Computing embeddings and metrics...")
            embs_np = trainer.compute_embeddings_for_model(model_diversity)
            metrics = trainer.evaluate_embeddings(embs_np)

            results.append(
                {
                    "repeat_idx": repeat_idx,
                    "repeat_number": repeat_idx + 1,
                    "seed": repeat_seed,
                    "n_models": n_models,
                    "loaded_from_checkpoint": loaded_from_checkpoint,
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": metrics,
                }
            )

    aggregated_results = aggregate_results_by_train_size(results)

    print("\nGenerating plots...")
    plot_correlation_vs_samples(aggregated_results, plots_dir)
    plot_recall_vs_samples(aggregated_results, plots_dir)
    save_results(output_base, results)
    print_summary(aggregated_results)


if __name__ == "__main__":
    main()
