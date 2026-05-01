import argparse
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
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Добавляем путь к корню проекта для импорта модулей
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from dependencies.GCN import (
    GAT_ver_2,
    CustomDataset,
    collate_graphs,
    train_model_accuracy,
)
from dependencies.data_generator import load_dataset
from dependencies.train_config import TrainConfig

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
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "ablation_study/accuracy_ablations/results"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "ablation_study/accuracy_ablations/correlation_analysis"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "ablation_study/accuracy_ablations/model_cache.npz"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Accuracy surrogate ablation study")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
    )
    parser.add_argument(
        "--model_counts",
        type=int,
        nargs="+",
        default=[0, 2000, 1500, 1000, 500, 250, 750],
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
        help="Directory for accuracy checkpoints and JSON results",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for plots",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help="Path to cached model metadata (.npz)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached model metadata if available",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of repeated runs for each training set size",
    )
    return parser.parse_args()


class AccuracyAblationTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dataset_path = Path(config.dataset_path)
        self.all_model_paths: List[Path] = []
        self.all_accuracies: Optional[np.ndarray] = None
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []

    def load_all_models(self, cache_path: Optional[Path] = None, use_cache: bool = False) -> None:
        cache_path = Path(cache_path) if cache_path else None

        if use_cache and cache_path and cache_path.exists():
            print(f"Loading model metadata from cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            self.all_model_paths = [Path(p) for p in cached["paths"]]
            self.all_accuracies = cached["accuracies"].astype(float)
            print(f"Loaded {len(self.all_model_paths)} models from cache")
            return

        load_dataset(self.config)
        self.all_model_paths = sorted(self.dataset_path.rglob("*.json"))
        self.config.models_dict_path = list(self.all_model_paths)
        self.config.acc_models_dict_path = list(self.all_model_paths)

        print(f"Total models available: {len(self.all_model_paths)}")
        accuracies = []

        for arch_json_path in tqdm(self.all_model_paths, desc="Loading pretrained archs"):
            arch = json.loads(arch_json_path.read_text(encoding="utf-8"))
            accuracies.append(arch.get("valid_accuracy", arch.get("test_accuracy", 0.0)) * 100)

        self.all_accuracies = np.array(accuracies, dtype=float)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                paths=np.array([str(p) for p in self.all_model_paths], dtype=object),
                accuracies=self.all_accuracies,
            )
            print(f"Saved model metadata cache to: {cache_path}")

    def prepare_common_data(self, random_seed: int = 42) -> None:
        indices = np.arange(len(self.all_model_paths))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.seed(None)

        train_size = int(0.8 * len(indices))
        self.train_indices = list(indices[:train_size])
        self.test_indices = list(indices[train_size:])
        print(f"Train pool size: {len(self.train_indices)}")
        print(f"Test pool size: {len(self.test_indices)}")

    def create_subset_dataset(self, indices: List[int], preload: bool = False):
        subset_paths = [self.all_model_paths[i] for i in indices]
        subset_accs = self.all_accuracies[indices]
        return CustomDataset(subset_paths, subset_accs, preload=preload)

    def create_accuracy_dataloaders(self, indices: List[int]):
        subset_dataset = self.create_subset_dataset(indices, preload=True)

        valid_count = max(1, int(round((1 - self.config.train_size) * len(subset_dataset))))
        train_count = len(subset_dataset) - valid_count
        if train_count <= 0:
            train_count = len(subset_dataset) - 1
            valid_count = 1

        train_dataset, valid_dataset = torch.utils.data.random_split(
            subset_dataset,
            [train_count, valid_count],
        )

        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": False,
            "collate_fn": collate_graphs,
        }


        train_loader = DataLoader(
            train_dataset,
            batch_size=min(8, self.config.batch_size),
            shuffle=True,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1024,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, valid_loader

    def build_accuracy_model(self):
        return GAT_ver_2(
            input_dim=self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
            output_activation="none",
            pre_norm=True,
            pooling="attn",
        ).to(self.device)

    def train_accuracy_model(self, indices: List[int]):
        model = self.build_accuracy_model()
        train_loader, valid_loader = self.create_accuracy_dataloaders(indices)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.acc_lr_start)
        train_model_accuracy(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            num_epochs=self.config.acc_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.acc_lr_end,
        )
        model.eval()
        return model

    def load_accuracy_model(self, checkpoint_path: Path):
        model = self.build_accuracy_model()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def save_accuracy_model(model, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model_accuracy.pth")

    def predict_accuracies(self, model_accuracy) -> np.ndarray:
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

        predictions = np.zeros(len(self.all_model_paths), dtype=float)
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                batch_predictions = (
                    model_accuracy(batch.x, batch.edge_index, batch.batch)
                    .view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                batch_size = len(batch_predictions)
                predictions[cursor:cursor + batch_size] = batch_predictions
                cursor += batch_size

        if cursor != len(self.all_model_paths):
            raise ValueError(
                "Prediction extraction length mismatch: "
                f"predicted={cursor}, expected={len(self.all_model_paths)}"
            )

        return predictions

    def evaluate_predictions(self, predicted_accuracies: np.ndarray):
        test_indices = np.array(self.test_indices)
        true_accuracies = self.all_accuracies[test_indices]
        predicted_test_accuracies = predicted_accuracies[test_indices]

        spearman_corr, spearman_p = spearmanr(predicted_test_accuracies, true_accuracies)
        kendall_corr, kendall_p = kendalltau(predicted_test_accuracies, true_accuracies)
        recall_metrics = compute_recall_at_k_for_accuracy(
            predicted_accuracies,
            self.all_accuracies,
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



def compute_recall_at_k_for_accuracy(
    predicted_accuracies,
    true_accuracies,
    test_indices,
    k_values=(10, 50, 100),
):
    test_indices = np.array(test_indices)
    if len(test_indices) == 0:
        return {f"recall_at_{k}": np.nan for k in k_values}

    effective_k_values = [min(k, len(test_indices)) for k in k_values]

    true_test_accuracies = true_accuracies[test_indices]
    predicted_test_accuracies = predicted_accuracies[test_indices]

    true_order = test_indices[np.argsort(-true_test_accuracies)]
    predicted_order = test_indices[np.argsort(-predicted_test_accuracies)]

    result = {}
    for k, effective_k in zip(k_values, effective_k_values):
        true_topk = set(true_order[:effective_k])
        predicted_topk = set(predicted_order[:effective_k])
        recall_value = len(true_topk & predicted_topk) / effective_k
        result[f"recall_at_{k}"] = float(recall_value)
    return result



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



def plot_recall_vs_samples(aggregated_results: List[Dict], output_dir: Path):
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



def plot_correlation_vs_samples(aggregated_results: List[Dict], output_dir: Path):
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
        spearman_means,
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
        spearman_means - spearman_stds,
        spearman_means + spearman_stds,
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
    ax1.set_ylim(-1, 1)
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
    ax2.set_ylim(-1, 1)
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
    print("ACCURACY ABLATION SUMMARY")
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



def select_initial_indices(train_indices: List[int], n_models: int, seed: int) -> List[int]:
    if n_models == 0:
        return []
    np.random.seed(seed)
    selected = list(np.random.choice(train_indices, size=n_models, replace=False))
    np.random.seed(None)
    return selected



def extend_indices_by_true_accuracy(
    current_indices: List[int],
    train_indices: List[int],
    all_accuracies: np.ndarray,
    target_size: int,
) -> List[int]:
    if target_size <= len(current_indices):
        return list(current_indices)

    remaining_candidates = [idx for idx in train_indices if idx not in current_indices]
    remaining_candidates = sorted(
        remaining_candidates,
        key=lambda idx: all_accuracies[idx],
        reverse=True,
    )
    n_to_add = target_size - len(current_indices)
    return list(current_indices) + remaining_candidates[:n_to_add]



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

        trainer = AccuracyAblationTrainer(config)
        print("Loading all models...")
        trainer.load_all_models(
            cache_path=Path(args.cache_path),
            use_cache=args.use_cache,
        )
        trainer.prepare_common_data(random_seed=repeat_seed)

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
            checkpoint_path = save_dir / "model_accuracy.pth"

            print(f"\n{'=' * 60}")
            print(f"Processing {n_models} models (cumulative), repeat {repeat_idx + 1}/{args.num_repeats}")
            print(f"{'=' * 60}")

            if iteration_idx == 0:
                cumulative_indices = select_initial_indices(trainer.train_indices, n_models, repeat_seed)
            else:
                cumulative_indices = extend_indices_by_true_accuracy(
                    cumulative_indices,
                    trainer.train_indices,
                    trainer.all_accuracies,
                    n_models,
                )

            if args.skip_existing and checkpoint_path.exists():
                print(f"Loading accuracy surrogate from {checkpoint_path}...")
                model_accuracy = trainer.load_accuracy_model(checkpoint_path)
                loaded_from_checkpoint = True
            else:
                if n_models == 0:
                    print("Initializing accuracy surrogate with random weights (no training)...")
                    model_accuracy = trainer.build_accuracy_model()
                else:
                    print(f"Training accuracy surrogate on {len(cumulative_indices)} models...")
                    model_accuracy = trainer.train_accuracy_model(cumulative_indices)
                trainer.save_accuracy_model(model_accuracy, save_dir)
                loaded_from_checkpoint = False

            print("Computing predictions and metrics...")
            predicted_accuracies = trainer.predict_accuracies(model_accuracy)
            metrics = trainer.evaluate_predictions(predicted_accuracies)

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
