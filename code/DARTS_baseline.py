import json
import os
import logging
from pathlib import Path
import re
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
from train_models import DatasetsInfo

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DartsArchitectureSelector(DiversityNESRunner):
    def __init__(self, config: TrainConfig):
        info = DatasetsInfo.get(config.dataset_name.lower())
        super().__init__(config, info)

    def get_best_models(self, train_loader, valid_loader):
        strategy = DartsStrategy(gradient_clip_val=5.)
        model_space = DartsSpace(
            width=self.config.width,
            num_cells=self.config.num_cells,
            dataset=self.dataset_key
        )

        devices_arg = "auto"
        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        tb_logger = TensorBoardLogger(save_dir="logs", name="deepens_darts")

        evaluator = Lightning(
            DartsClassificationModule(
                learning_rate=self.config.lr_start_final,
                weight_decay=3e-4,
                auxiliary_loss_weight=0.4,
                max_epochs=self.config.n_epochs_final,
                num_classes=self.num_classes,
                lr_final=self.config.lr_end_final,
                warmup_epochs=0
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

    def save_architectures(self, architectures):
        """Сохраняет архитектуры в JSON-файлы."""
        if not architectures:
            print("Warning: No architectures were selected, nothing to save.")
            return

        base_save_path = Path(self.config.best_models_save_path)
        base_save_path.mkdir(parents=True, exist_ok=True)
        
        existing_dirs = base_save_path.glob("models_json_*")
        indices = [int(re.search(r"(\d+)", p.name).group(1)) for p in existing_dirs if re.search(r"(\d+)", p.name)]
        next_index = max(indices, default=0) + 1
        
        arch_dir = base_save_path / f"models_json_{next_index}"
        arch_dir.mkdir()
        print(f"Saving DARTS architectures to: {arch_dir}")

        for i, arch in enumerate(architectures):
            file_name = f"model_{i+1}.json"
            file_path = arch_dir / file_name
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(arch, f, indent=4)
        
        print(f"Successfully saved {len(architectures)} DARTS architectures.")

    def run(self):
        self.config.selected_archs = []
        
        train_loader, valid_loader, _ = self.get_data_loaders(seed=self.config.seed)
        self.config.selected_archs.append(self.get_best_models(train_loader, valid_loader))
        self.config.selected_archs *= self.config.n_ensemble_models
        
        self.save_architectures(self.config.selected_archs)
        print("\nArchitecture search completed successfully!")
        print(f"Выбрано и сохранено {len(self.config.selected_archs)} архитектур.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARTS Architecture Selector")
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
    selector = DartsArchitectureSelector(config)
    selector.run()
