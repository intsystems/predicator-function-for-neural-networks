import json
import os
import shutil
from pathlib import Path

import argparse
import torch

from dependencies.train_config import TrainConfig
from train_models import DiversityNESRunner
from train_models import DatasetsInfo
from dependencies.data_generator import generate_arch_dicts

from tqdm import tqdm

NUM_PRETRAIN_EPOCHS = 5


class RandomSearchBaseline(DiversityNESRunner):
    def __init__(self, config: TrainConfig):
        info = DatasetsInfo.get(config.dataset_name.lower())
        super().__init__(config, info)

        self.pretrain_epochs = NUM_PRETRAIN_EPOCHS

    def run(self):
        self.config.selected_archs = []

        train_loader, valid_loader, test_loader = self.get_data_loaders()
        archs = generate_arch_dicts(self.config.n_models_to_generate)
        archs = [d["architecture"] for d in archs]

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
            self.models = []    # Don't need to save all models

        shutil.rmtree(Path(self.config.output_path) / "RandomSearch_tmp")
        self.config.n_epochs_final = n_epoch_final

        model_names = sorted(os.listdir(self.config.tmp_archs_path))

        accs = []
        for model_name in model_names:
            result_path = Path(self.config.tmp_archs_path) / model_name
            with open(result_path, "r") as f:
                data = json.load(f)
                accs.append(data["valid_accuracy"])
        
        _, best_model_idxs = torch.topk(torch.tensor(accs), self.config.n_ensemble_models)
        best_models_names = [model_names[i] for i in best_model_idxs]

        self.models = []
        for idx, model_name in enumerate(best_models_names):
            result_path = Path(self.config.tmp_archs_path) / model_name
            with open(result_path, "r") as f:
                data = json.load(f)
                arch = data["architecture"]
                self.train_model(
                    arch,
                    train_loader,
                    valid_loader,
                    idx,
                    save_dir_name="RandomSearch_models",
                    weight_init_seed=None,
                )
        shutil.rmtree(Path(self.config.tmp_archs_path))

        print("\nEvaluating ensemble...")
        stats = self.collect_ensemble_stats(test_loader)
        self.finalize_ensemble_evaluation(stats, "RandomSearch_baseline_results")

        print("\nAll models processed successfully!")
        print(f"Выбрано и сохранено {len(best_models_names)} моделей.")

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
