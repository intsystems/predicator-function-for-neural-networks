import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(1, "../dependencies")

from dependencies.GCN import (
    GAT_ver_2,
    CustomDataset,
    collate_graphs,
    train_model_accuracy,
)
from dependencies.train_config import TrainConfig


DEFAULT_OUTPUT_BASE = (
    "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/"
    "results/surrogates/ablation"
)
DEFAULT_CACHE_FILE = (
    "/home/alexander/RAS/m1p/predicator-function-for-neural-networks/code/"
    "results/surrogates/ablation/model_cache.npz"
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ablation Study Surrogate Trainer")
    parser.add_argument("--hyperparameters_json", type=str, required=True)
    parser.add_argument(
        "--step_size",
        type=int,
        default=100,
        help="Number of models to add each iteration",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=None,
        help="Maximum models to train up to (default: all)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip iterations where models already exist",
    )
    parser.add_argument(
        "--use_cache", action="store_true", default=True, help="Use cached model data"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help="Base output directory",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: output_base)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["random"],
        choices=["random", "greedy", "ucb_bayes"],
        help="Selection strategies to run",
    )
    parser.add_argument(
        "--ucb_passes",
        type=int,
        default=20,
        help="Number of forward passes for UCB-Bayes strategy (default: 20)",
    )
    parser.add_argument(
        "--ucb_beta",
        type=float,
        default=1.0,
        help="Exploration parameter for UCB-Bayes strategy (default: 1.0)",
    )
    return parser.parse_args()


class SelectionStrategy(Protocol):
    def select_initial(
        self, trainer, n_models: int, train_indices: List[int]
    ) -> List[int]:
        """Select initial models for training."""
        ...

    def select_next(
        self,
        trainer,
        current_train_indices: List[int],
        candidate_indices: List[int],
        n_select: int,
        acc_model,
        device,
    ) -> List[int]:
        """Select next models to add to training set."""
        ...


class RandomSelectionStrategy:
    def select_initial(
        self, trainer, n_models: int, train_indices: List[int]
    ) -> List[int]:
        np.random.seed(42)
        selected = np.random.choice(train_indices, size=n_models, replace=False)
        return list(selected)

    def select_next(
        self,
        trainer,
        current_train_indices: List[int],
        candidate_indices: List[int],
        n_select: int,
        acc_model,
        device,
    ) -> List[int]:
        np.random.seed(None)
        selected = np.random.choice(candidate_indices, size=n_select, replace=False)
        return list(selected)


class GreedySelectionStrategy:
    def __init__(self, max_sample_size: int = 50):
        self.max_sample_size = max_sample_size

    def select_initial(
        self, trainer, n_models: int, train_indices: List[int]
    ) -> List[int]:
        np.random.seed(42)
        selected = np.random.choice(train_indices, size=n_models, replace=False)
        return list(selected)

    def select_next(
        self,
        trainer,
        current_train_indices: List[int],
        candidate_indices: List[int],
        n_select: int,
        acc_model,
        device,
    ) -> List[int]:
        if len(candidate_indices) <= n_select:
            return candidate_indices

        acc_model.eval()

        candidate_paths = [trainer.all_model_paths[i] for i in candidate_indices]
        candidate_accs_arr = trainer.all_accuracies[candidate_indices]
        candidate_ds = CustomDataset(candidate_paths, candidate_accs_arr)
        candidate_loader = DataLoader(
            candidate_ds,
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graphs,
        )

        with torch.no_grad():
            pred_accs = []
            for data in candidate_loader:
                data = data.to(device)
                pred = (
                    acc_model(data.x, data.edge_index, data.batch)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                pred_accs.append(pred)

        candidate_pred_accs = np.concatenate(pred_accs)

        selected_indices = []
        remaining_candidates = list(candidate_indices)
        remaining_pred_accs = candidate_pred_accs.copy()

        best_first_idx = np.argmax(remaining_pred_accs)
        selected_indices.append(remaining_candidates.pop(best_first_idx))
        remaining_pred_accs = np.delete(remaining_pred_accs, best_first_idx)

        while len(selected_indices) < n_select and len(remaining_candidates) > 0:
            best_idx = np.argmax(remaining_pred_accs)
            selected_indices.append(remaining_candidates.pop(best_idx))
            remaining_pred_accs = np.delete(remaining_pred_accs, best_idx)

        return selected_indices


class UCBBayesSelectionStrategy:
    def __init__(self, T: int = 20, beta: float = 1.0, max_sample_size: int = 1):
        self.T = T
        self.beta = beta

    def select_initial(
        self, trainer, n_models: int, train_indices: List[int]
    ) -> List[int]:
        np.random.seed(42)
        selected = np.random.choice(train_indices, size=n_models, replace=False)
        return list(selected)

    def select_next(
        self,
        trainer,
        current_train_indices: List[int],
        candidate_indices: List[int],
        n_select: int,
        acc_model,
        device,
    ) -> List[int]:
        if len(candidate_indices) <= n_select:
            return candidate_indices

        acc_model.train()

        candidate_paths = [trainer.all_model_paths[i] for i in candidate_indices]
        candidate_accs_arr = trainer.all_accuracies[candidate_indices]
        candidate_ds = CustomDataset(candidate_paths, candidate_accs_arr)
        candidate_loader = DataLoader(
            candidate_ds,
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graphs,
        )

        all_predictions = []

        with torch.no_grad():
            for _ in range(self.T):
                batch_predictions = []
                for data in candidate_loader:
                    data = data.to(device)
                    pred = (
                        acc_model(data.x, data.edge_index, data.batch)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    batch_predictions.append(pred)

                if batch_predictions:
                    all_predictions.append(np.concatenate(batch_predictions))
                else:
                    all_predictions.append(np.array([]))

        if all_predictions and len(all_predictions[0]) > 0:
            predictions_array = np.array(all_predictions)
            candidate_means = np.mean(predictions_array, axis=0)
            candidate_stds = np.std(predictions_array, axis=0)

            candidate_ucbs = candidate_means + self.beta * candidate_stds
        else:
            candidate_ucbs = np.zeros(len(candidate_indices))

        selected_indices = []
        remaining_candidates = list(candidate_indices)
        remaining_ucbs = candidate_ucbs.copy()

        while len(selected_indices) < n_select and len(remaining_candidates) > 0:
            best_idx = np.argmax(remaining_ucbs)
            selected_indices.append(remaining_candidates.pop(best_idx))
            remaining_ucbs = np.delete(remaining_ucbs, best_idx)

        return selected_indices


STRATEGIES = {
    "random": RandomSelectionStrategy,
    "greedy": GreedySelectionStrategy,
    "ucb_bayes": UCBBayesSelectionStrategy,
}


class AblationSurrogateTrainer:
    def __init__(self, config: TrainConfig, cache_file: str = None):
        self.config = config
        self.device = torch.device(config.device)
        self.dataset_path = Path(config.dataset_path)
        self.all_predictions = None
        self.all_accuracies = None
        self.all_model_paths = None
        self.cache_file = cache_file

    def load_all_models(self):
        cache_path = Path(self.cache_file) if self.cache_file else None
        if cache_path and cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.all_predictions = data["predictions"]
            self.all_accuracies = data["accuracies"]
            self.all_model_paths = [Path(p) for p in data["paths"]]
            print(f"Loaded {len(self.all_model_paths)} models from cache")
            return

        all_json_files = sorted(Path(self.config.acc_dataset_path).rglob("*.json"))
        self.all_model_paths = all_json_files
        print(f"Total models available: {len(all_json_files)}")

        print("Loading all predictions and accuracies...")
        all_preds = []
        all_accs = []

        for path in tqdm(self.all_model_paths, desc="Loading model data"):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            prediction_name = (
                "valid_predictions"
                if "valid_predictions" in data
                else "test_predictions"
            )
            all_preds.append(np.array(data[prediction_name]))

            acc = data.get("valid_accuracy", data.get("test_accuracy", 0))
            all_accs.append(acc)

        self.all_predictions = np.array(all_preds)
        self.all_accuracies = np.array(all_accs) * 100

        self.all_model_paths = [Path(p) for p in all_json_files]
        print(f"Loaded {len(self.all_model_paths)} models")

        if cache_path:
            np.savez(
                cache_path,
                predictions=self.all_predictions,
                accuracies=self.all_accuracies,
                paths=[str(p) for p in self.all_model_paths],
            )
            print(f"Cached to {cache_path}")

    def create_subset_dataset(self, indices: List[int]):
        subset_paths = [self.all_model_paths[i] for i in indices]
        subset_accs = self.all_accuracies[indices]
        return CustomDataset(subset_paths, subset_accs)

    def create_dataloaders(self, base_dataset):
        cfg = self.config

        train_size = int(cfg.train_size * len(base_dataset))
        valid_size = len(base_dataset) - train_size
        train_ds, valid_ds = random_split(base_dataset, [train_size, valid_size])

        cfg.train_loader_accuracy = DataLoader(
            train_ds,
            batch_size=min(8, cfg.batch_size),
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_graphs,
        )
        cfg.valid_loader_accuracy = DataLoader(
            valid_ds,
            batch_size=1024,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_graphs,
        )

        return cfg.train_loader_accuracy, cfg.valid_loader_accuracy

    def build_accuracy_model(self):
        model = GAT_ver_2(
            input_dim=self.config.input_dim,
            output_dim=1,
            dropout=self.config.acc_dropout,
            heads=self.config.acc_n_heads,
            output_activation="none",
            pre_norm=True,
            pooling="attn",
        )
        return model

    def train_accuracy_model(self, train_loader, valid_loader):
        model = self.build_accuracy_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.acc_lr_start)

        train_model_accuracy(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=opt,
            criterion=nn.MSELoss(),
            num_epochs=self.config.acc_num_epochs,
            device=self.device,
            developer_mode=self.config.developer_mode,
            final_lr=self.config.acc_lr_end,
        )
        return model

    def load_accuracy_model(self, checkpoint_path: Path):
        model = self.build_accuracy_model()
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        return model

    def evaluate_dataset(self, train_indices: List[int]):
        """Calculate metrics for current training dataset."""
        n_models = len(train_indices)
        if n_models == 0:
            return {
                "mean_accuracy": 0.0,
                "top_k_overlap": 0.0,
                "top_100_fraction": 0.0,
                "n_models": 0,
            }

        ground_truth_accs = self.all_accuracies[train_indices]
        mean_accuracy = float(np.mean(ground_truth_accs))

        all_accs_sorted_indices = np.argsort(self.all_accuracies)[::-1]
        top_k_indices = set(all_accs_sorted_indices[:n_models])
        current_set = set(train_indices)

        overlap = len(top_k_indices & current_set)
        top_k_overlap = overlap / n_models

        top_100_indices = set(all_accs_sorted_indices[:100])
        top_100_fraction = len(top_100_indices & current_set) / n_models

        return {
            "mean_accuracy": mean_accuracy,
            "top_k_overlap": top_k_overlap,
            "top_100_fraction": top_100_fraction,
            "n_models": n_models,
        }

    @staticmethod
    def save_model(acc_model, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(acc_model.state_dict(), save_dir / "model_accuracy.pth")


def extract_metrics_by_strategy(
    results_by_strategy: Dict[str, List[Dict[str, Any]]], metrics_key: str = "metrics"
):
    """Extract metrics grouped by strategy."""
    metrics = {
        "n_models": {},
        "mean_accuracy": {},
        "top_k_overlap": {},
        "top_100_fraction": {},
    }

    for strategy_name, results in results_by_strategy.items():
        metrics["n_models"][strategy_name] = [r["n_models"] for r in results]
        metrics["mean_accuracy"][strategy_name] = [
            r.get(metrics_key, r.get("metrics", {}))["mean_accuracy"] for r in results
        ]
        metrics["top_k_overlap"][strategy_name] = [
            r.get(metrics_key, r.get("metrics", {}))["top_k_overlap"] for r in results
        ]
        metrics["top_100_fraction"][strategy_name] = [
            r.get(metrics_key, r.get("metrics", {}))["top_100_fraction"] for r in results
        ]

    return metrics


def _plot_metric(metrics, metric_key, ylabel, title, plots_dir: Path, file_stem):
    """Render a single metric plot comparing all strategies."""
    strategies = list(metrics["n_models"].keys())
    colors = plt.cm.tab10.colors[: len(strategies)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, color in zip(strategies, colors):
        ax.plot(
            metrics["n_models"][strategy],
            metrics[metric_key][strategy],
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=strategy,
        )
    ax.set_xlabel("Number of Models", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{file_stem}.pdf", dpi=150)
    plt.savefig(plots_dir / f"{file_stem}.png", dpi=150)
    plt.close()


def plot_combined_metrics(results_by_strategy, plots_dir: Path):
    """Generate combined comparison plots for all strategies."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "metrics",
            "mean_accuracy",
            "Mean Accuracy",
            "Mean Accuracy vs Number of Models",
            "mean_accuracy",
        ),
        (
            "metrics",
            "top_k_overlap",
            "Top-K Overlap",
            "Top-K Overlap vs Number of Models",
            "top_k_overlap",
        ),
        (
            "metrics",
            "top_100_fraction",
            "Top-100 Fraction",
            "Top-100 Fraction vs Number of Models",
            "top_100_fraction",
        ),
        (
            "metrics_recent",
            "mean_accuracy",
            "Mean Accuracy (Last Added Models)",
            "Mean Accuracy of Last Added Models vs Number of Models",
            "mean_accuracy_recent",
        ),
        (
            "metrics_recent",
            "top_k_overlap",
            "Top-K Overlap (Last Added Models)",
            "Top-K Overlap of Last Added Models vs Number of Models",
            "top_k_overlap_recent",
        ),
        (
            "metrics_recent",
            "top_100_fraction",
            "Top-100 Fraction (Last Added Models)",
            "Top-100 Fraction of Last Added Models vs Number of Models",
            "top_100_fraction_recent",
        ),
    ]

    for metrics_key, metric_name, ylabel, title, file_stem in specs:
        grouped_metrics = extract_metrics_by_strategy(results_by_strategy, metrics_key)
        _plot_metric(grouped_metrics, metric_name, ylabel, title, plots_dir, file_stem)

    print(f"Combined plots saved to {plots_dir}")


def _build_model_counts(total_models: int, step_size: int, max_models: Optional[int]):
    max_count = min(max_models or total_models, total_models)
    if max_count <= 0:
        return []

    model_counts = list(range(step_size, max_count + 1, step_size))
    if not model_counts or model_counts[-1] != max_count:
        model_counts.append(max_count)

    return model_counts


def _create_strategy(strategy_name: str, args):
    strategy_class = STRATEGIES[strategy_name]
    if strategy_name == "ucb_bayes":
        return strategy_class(T=args.ucb_passes, beta=args.ucb_beta)
    return strategy_class()


def _select_iteration_indices(
    strategy,
    trainer,
    train_indices: List[int],
    current_train_indices: List[int],
    n_models: int,
    step_size: int,
    acc_model,
    iteration_idx: int,
) -> Tuple[List[int], List[int]]:
    if iteration_idx == 0:
        current_train_indices = strategy.select_initial(trainer, n_models, train_indices)
        recent_indices = list(current_train_indices)
    else:
        candidates = [i for i in train_indices if i not in current_train_indices]
        new_indices = strategy.select_next(
            trainer,
            current_train_indices,
            candidates,
            step_size,
            acc_model,
            trainer.device,
        )
        current_train_indices = current_train_indices + new_indices
        recent_indices = list(new_indices)

    return current_train_indices, recent_indices


def _prepare_accuracy_model(
    strategy_name: str,
    trainer,
    current_train_indices,
    n_models: int,
    checkpoint_path: Path,
    skip_existing: bool,
):
    if strategy_name == "random":
        return None, False

    if skip_existing and checkpoint_path.exists():
        print(f"Loading accuracy surrogate from {checkpoint_path}...")
        return trainer.load_accuracy_model(checkpoint_path), True

    base_ds = trainer.create_subset_dataset(current_train_indices)
    train_acc_loader, valid_acc_loader = trainer.create_dataloaders(base_ds)

    print(f"Training accuracy surrogate on {n_models} models...")
    return trainer.train_accuracy_model(train_acc_loader, valid_acc_loader), False


def _print_metrics(prefix: str, metrics: Dict[str, float]):
    print(
        f"{prefix}: Mean Accuracy={metrics['mean_accuracy']:.2f}, "
        f"Top-K Overlap={metrics['top_k_overlap']:.4f}, "
        f"Top-100 Fraction={metrics['top_100_fraction']:.4f}"
    )


def run_strategy(
    trainer,
    strategy,
    strategy_name,
    step_size,
    args,
    output_base,
):
    """Run ablation study for a single strategy."""
    train_indices = list(range(len(trainer.all_model_paths)))
    model_counts = _build_model_counts(len(train_indices), step_size, args.max_models)

    if not model_counts:
        print(f"No model counts to process for strategy {strategy_name}.")
        return []

    strategy_dir = Path(output_base) / strategy_name

    results = []
    current_train_indices = []
    acc_model = None

    for iteration_idx, n_models in enumerate(model_counts):
        save_dir = strategy_dir / f"models_{n_models}"
        checkpoint_path = save_dir / "model_accuracy.pth"

        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy_name} | Processing {n_models} models (cumulative)")
        print(f"{'=' * 60}")

        current_train_indices, recent_indices = _select_iteration_indices(
            strategy,
            trainer,
            train_indices,
            current_train_indices,
            n_models,
            step_size,
            acc_model,
            iteration_idx,
        )

        acc_model, loaded_from_checkpoint = _prepare_accuracy_model(
            strategy_name,
            trainer,
            current_train_indices,
            n_models,
            checkpoint_path,
            args.skip_existing,
        )

        print(f"Evaluating dataset on {n_models} models...")
        metrics = trainer.evaluate_dataset(current_train_indices)
        recent_metrics = trainer.evaluate_dataset(recent_indices)

        _print_metrics("Cumulative metrics", metrics)
        _print_metrics("Last-added metrics", recent_metrics)

        if strategy_name != "random" and not loaded_from_checkpoint:
            print(f"Saving models to {save_dir}...")
            trainer.save_model(acc_model, save_dir)

        results.append(
            {
                "n_models": n_models,
                "n_recent_models": len(recent_indices),
                "metrics": metrics,
                "metrics_recent": recent_metrics,
            }
        )

    return results


def save_results(output_base: str, results_by_strategy: Dict[str, List[Dict[str, Any]]]):
    for strategy_name, results in results_by_strategy.items():
        results_path = Path(output_base) / strategy_name / "ablation_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def print_summary(results_by_strategy: Dict[str, List[Dict[str, Any]]]):
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    for strategy_name, results in results_by_strategy.items():
        print(f"\nStrategy: {strategy_name}")
        if not results:
            print("  No results collected.")
            continue

        for result in results:
            cumulative = result["metrics"]
            recent = result.get("metrics_recent", cumulative)
            print(
                f"  n_models={result['n_models']:4d}: "
                f"Cumulative Mean Acc={cumulative['mean_accuracy']:6.2f}, "
                f"Cumulative Top-K Overlap={cumulative['top_k_overlap']:.4f}, "
                f"Recent Mean Acc={recent['mean_accuracy']:6.2f}, "
                f"Recent Top-K Overlap={recent['top_k_overlap']:.4f}"
            )


def main():
    """Main entry point for ablation study."""
    args = parse_arguments()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    config = TrainConfig(**params)

    cache_file = DEFAULT_CACHE_FILE if args.use_cache else None
    trainer = AblationSurrogateTrainer(config, cache_file=cache_file)

    print("Loading all models...")
    trainer.load_all_models()

    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.output_base)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_by_strategy = {}
    for strategy_name in args.strategies:
        strategy = _create_strategy(strategy_name, args)

        print(f"\n{'#' * 60}")
        print(f"# Running strategy: {strategy_name}")
        print(f"{'#' * 60}")

        results_by_strategy[strategy_name] = run_strategy(
            trainer,
            strategy,
            strategy_name,
            args.step_size,
            args,
            args.output_base,
        )

    print("\nGenerating combined plots...")
    plot_combined_metrics(results_by_strategy, plots_dir)

    save_results(args.output_base, results_by_strategy)
    print_summary(results_by_strategy)


if __name__ == "__main__":
    main()
