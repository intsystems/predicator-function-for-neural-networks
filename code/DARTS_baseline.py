import json
import os
import logging
from pathlib import Path
import re
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

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices_arg = [7,6,5,3]

        print(f"🔍 Developer mode from config: {self.config.developer_mode}")
        print(f"🔍 Train batches: {len(train_loader)}")
        print(f"🔍 Val batches: {len(valid_loader)}")
        
        tb_logger = TensorBoardLogger(
            save_dir="logs", 
            name=f"deepens_darts_{self.config.dataset_name}"
        )

        module = DartsClassificationModule(
            learning_rate=self.config.lr_start_final,
            weight_decay=self.config.weight_decay,
            auxiliary_loss_weight=self.config.auxiliary_loss_weight,
            max_epochs=self.config.n_epochs_final,
            num_classes=self.num_classes,
            lr_final=self.config.lr_end_final,
            warmup_epochs=0
        )
        
        def patched_get_result(self):
            if 'val_acc' in self.trainer.callback_metrics:
                return self.trainer.callback_metrics['val_acc'].item()
            elif 'train_acc' in self.trainer.callback_metrics:
                return self.trainer.callback_metrics['train_acc'].item()
            else:
                return 0.0
        
        module._get_result_for_report = lambda: patched_get_result(module)

        evaluator = Lightning(
            module,
            trainer=Trainer(
                max_epochs=self.config.n_epochs_final,
                fast_dev_run=self.config.developer_mode,
                accelerator=accelerator,
                devices=devices_arg,
                enable_progress_bar=True,
                logger=tb_logger,
            ),
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        experiment = NasExperiment(model_space, evaluator, strategy)
        experiment.run()
        
        top_models = experiment.export_top_models(formatter='dict')
        if not top_models:
            raise RuntimeError("DARTS experiment did not produce any models")
        
        experiment.stop()
        return top_models[0]


    def save_architectures(self, architectures):
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
        best_arch = self.get_best_models(train_loader, valid_loader)
        
        # Если нужно несколько копий одной архитектуры (для разных seed):
        self.config.selected_archs = [best_arch] * self.config.n_ensemble_models
        
        # Если нужно топ-N разных архитектур — используй export_top_models(top_k=N)
        
        self.save_architectures(self.config.selected_archs)
        print("\nArchitecture search completed successfully!")
        print(f"Выбрано и сохранено {len(self.config.selected_archs)} архитектур.")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
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