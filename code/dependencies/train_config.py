from dataclasses import dataclass, field
from typing import Optional, Any
import random
import torch
import numpy as np


@dataclass
class TrainConfig:
    dataset_path: str
    device: str
    developer_mode: bool
    n_models: int
    upper_margin: float
    lower_margin: float
    diversity_matrix_metric: str
    train_size: float
    batch_size: int
    acc_num_epochs: int
    acc_lr: float
    acc_final_lr: float
    acc_dropout: float
    acc_n_heads: int
    div_num_epochs: int
    div_lr: float
    div_final_lr: float
    div_dropout: float
    div_n_heads: int
    margin: float
    div_output_dim: int
    surrogate_inference_path: str
    input_dim: int

    n_ensemble_models: int
    n_models_in_pool: int
    n_models_to_generate: int
    batch_size_inference: int
    min_accuracy_for_pool: float
    
    tmp_archs_path: str
    best_models_save_path: str

    prepared_dataset_path: str
    evaluate_ensemble_flag: bool

    random_choice_out_of_best: bool

    n_epochs_final: int
    lr_final: float
    batch_size_final: int
    dataset_name: str
    final_dataset_path: str
    n_models_to_evaluate: int
    output_path: str
    width: int
    num_cells: int
    num_workers: int
    n_ece_bins: int

    seed: Optional[int] = None

    # Internal fields
    model_accuracy: Optional[Any] = field(default=None, init=False)
    model_diversity: Optional[Any] = field(default=None, init=False)
    models_dict_path: list = field(default_factory=list, init=False)
    diversity_matrix: Optional[np.ndarray] = field(default=None, init=False)
    discrete_diversity_matrix: Optional[np.ndarray] = field(default=None, init=False)
    base_train_dataset: Optional[Any] = field(default=None, init=False)
    base_valid_dataset: Optional[Any] = field(default=None, init=False)
    train_dataset: Optional[Any] = field(default=None, init=False)
    valid_dataset: Optional[Any] = field(default=None, init=False)
    full_triplet_dataset: Optional[Any] = field(default=None, init=False)
    train_loader_diversity: Optional[Any] = field(default=None, init=False)
    valid_loader_diversity: Optional[Any] = field(default=None, init=False)
    train_loader_accuracy: Optional[Any] = field(default=None, init=False)
    valid_loader_accuracy: Optional[Any] = field(default=None, init=False)

    best_models: list = field(default_factory=list, init=False)
    best_embeddings: list = field(default_factory=list, init=False)
    potential_archs: list = field(default_factory=list, init=False)
    potential_embeddings: list = field(default_factory=list, init=False)
    potential_accuracies: list = field(default_factory=list, init=False)
    selected_archs: list = field(default_factory=list, init=False)
    selected_embs: list = field(default_factory=list, init=False)
    selected_accs: list = field(default_factory=list, init=False)
    selected_indices: list = field(default_factory=list, init=False)

    def __post_init__(self):
        assert (
            0 <= self.lower_margin < self.upper_margin <= 1
        ), "Margins must satisfy 0 ≤ lower < upper ≤ 1"
        assert 0 <= self.train_size <= 1, "train_size must be in [0,1]"
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
