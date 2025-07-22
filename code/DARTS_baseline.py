import json
import shutil
import os
import logging
from pathlib import Path
import shutil

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

from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepEnsBaseline(DiversityNESRunner):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

        dataset_map = {
            "cifar10": "cifar",
            "cifar100": "cifar100",
            "fashionmnist": "cifar"
        }

        if self.config.dataset_name.lower() not in dataset_map:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
        self.dataset = dataset_map[self.config.dataset_name.lower()]

        self.num_classes = {
            "cifar10": 10,
            "cifar100": 100,
            "fashionmnist": 10
        }[self.config.dataset_name.lower()]
    
    def get_best_models(self, train_loader, valid_loader):
        strategy = DartsStrategy(gradient_clip_val=5.)
        model_space = DartsSpace(
            width=self.config.width,           # the initial filters (channel number) for the model
            num_cells=self.config.num_cells,        # the number of stacked cells in total
            dataset=self.dataset
        )

        devices_arg = 1 if self.device.type == "cuda" else "auto"
        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        tb_logger = TensorBoardLogger(save_dir="logs", name="deepens_darts")

        evaluator = Lightning(
            DartsClassificationModule(
                learning_rate=self.config.lr_final,
                weight_decay=3e-4,
                auxiliary_loss_weight=0.4,
                max_epochs=self.config.n_epochs_final,
                num_classes=self.num_classes,
            ),
            trainer=Trainer(
                max_epochs=self.config.n_epochs_final,
                fast_dev_run=self.config.developer_mode,
                accelerator=accelerator,
                devices=devices_arg,
                enable_progress_bar=True,
                logger=tb_logger
            ),
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        experiment = NasExperiment(model_space, evaluator, strategy)
        experiment.run()
        return experiment.export_top_models(formatter='dict')[0]

    def run(self):
        self.config.selected_archs = []
        shutil.rmtree(self.config.output_path, ignore_errors=True)

        for _ in tqdm(range(self.config.n_ensemble_models), desc="Finding best architectures"):
            train_loader, valid_loader, test_loader = self.get_data_loaders()
            self.config.selected_archs.append(self.get_best_models(train_loader, valid_loader))
        
        for idx, arch in enumerate(tqdm(self.config.selected_archs, desc="Training models")):
            print(f"\nTraining model {idx+1}/{len(self.config.selected_archs)}")
            self.train_model(arch, train_loader, valid_loader, idx, save_dir_name="DARTS_baseline_models")

        print("\nEvaluating ensemble...")
        stats = self.collect_ensemble_stats(test_loader)
        self.finalize_ensemble_evaluation(stats, "DARTS_baseline_results.txt")

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

    config.evaluate_ensemble_flag = False #need for get validation loader\
    runner = DeepEnsBaseline(config)
    runner.run()
    