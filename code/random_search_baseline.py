import json
import os
import shutil
import logging
from pathlib import Path

import argparse
import torch
from nni.nas.strategy import DARTS as DartsStrategy
from nni.nas.experiment import NasExperiment
from nni.nas.evaluator.pytorch import Lightning, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dependencies.train_config import TrainConfig
from train_models import DiversityNESRunner
from dependencies.darts_classification_module import DartsClassificationModule
from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.data_generator import generate_arch_dicts

from tqdm import tqdm

NUM_PRETRAIN_EPOCHS = 5


class RandomSearchBaseline(DiversityNESRunner):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

        self.pretrain_epochs = NUM_PRETRAIN_EPOCHS

    def run(self):
        self.config.selected_archs = []

        train_loader, valid_loader, test_loader = self.get_data_loaders()
        archs = generate_arch_dicts(self.config.n_models_to_generate)

        shutil.rmtree(Path(self.config.tmp_archs_path), ignore_errors=True)
        Path(self.config.tmp_archs_path).mkdir(parents=True, exist_ok=True)

        n_epoch_final = self.config.n_epochs_final
        self.config.n_epochs_final = self.pretrain_epochs
        for idx, arch in enumerate(tqdm(archs, desc="Training models")):
            print(f"\nTraining model {idx+1}/{len(archs)}")
            model = self.train_model(
                arch,
                train_loader,
                valid_loader,
                idx,
                save_dir_name=f"RandomSearch_tmp",
                weight_init_seed=None,
            )
            self.evaluate_and_save_results(
                model,
                arch,
                valid_loader,
                self.config.tmp_archs_path,
            )
            self.models = []    # Doesn't need to save models in prepare dataset mode

        shutil.rmtree(Path(self.config.output_path) / "RandomSearch_tmp")
        self.config.n_epochs_final = n_epoch_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RandomSearch NAS Runner")
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
    runner = RandomSearchBaseline(config)
    runner.run()
