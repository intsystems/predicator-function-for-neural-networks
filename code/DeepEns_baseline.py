import json
import os
import logging
from pathlib import Path

import argparse
import torch
from nni.nas.strategy import DARTS as DartsStrategy
from nni.nas.experiment import NasExperiment
from nni.nas.evaluator.pytorch import Lightning, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dependencies.train_config import TrainConfig
from dependencies.darts_classification_module import DartsClassificationModule
from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from DARTS_baseline import DartsBaseline

from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepEnsBaseline(DartsBaseline):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

    def run(self):
        self.config.selected_archs = []

        train_loader, valid_loader, test_loader = self.get_data_loaders()
        self.config.selected_archs.append(
            self.get_best_models(train_loader, valid_loader)
        )
        self.config.selected_archs = (
            self.config.selected_archs * self.config.n_ensemble_models
        )

        for idx, arch in enumerate(
            tqdm(self.config.selected_archs, desc="Training models")
        ):
            print(f"\nTraining model {idx+1}/{len(self.config.selected_archs)}")
            self.train_model(
                arch,
                train_loader,
                valid_loader,
                idx,
                save_dir_name="DeepEns_baseline_models",
                weight_init_seed = self.config.seed + idx * 10
            )

        print("\nEvaluating ensemble...")
        stats = self.collect_ensemble_stats(test_loader)
        self.finalize_ensemble_evaluation(stats, "DeepEns_baseline_results")

        print("\nAll models processed successfully!")
        print(f"Выбрано и сохранено {len(self.config.selected_archs)} моделей.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTS NAS Runner")
    parser.add_argument(
        "--hyperparameters_json",
        type=str,
        required=True,
        help="Path to JSON file with TrainConfig parameters",
    )
    args = parser.parse_args()

    params = json.loads(Path(args.hyperparameters_json).read_text())
    params.update({"device": "cuda:0" if torch.cuda.is_available() else "cpu"})
    config = TrainConfig(**params)

    config.evaluate_ensemble_flag = False  # DARTS needs validation loader
    runner = DeepEnsBaseline(config)
    runner.run()
