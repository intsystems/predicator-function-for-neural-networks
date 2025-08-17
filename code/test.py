import os
import json
import shutil
import re
import argparse
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import nni
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from nni.nas.evaluator.pytorch import DataLoader, Lightning, Trainer
from nni.nas.space import model_context
from tqdm import tqdm

from utils_nni.DartsSpace import DARTS_with_CIFAR100 as DartsSpace
from dependencies.darts_classification_module import DartsClassificationModule
from dependencies.train_config import TrainConfig
from dependencies.data_generator import generate_arch_dicts


with open("datasets/cifar100_archs/model_1000.json", "r") as f:
    arch = json.load(f)
    arch = arch["architecture"]

dataset = "cifar100"
with model_context(arch):
    model = DartsSpace(
        width=4,
        num_cells=3,
        dataset=dataset,
        )

model.to("cpu")
model.load_state_dict(torch.load("datasets/cifar100_pth/model_1000.pth", map_location="cpu", weights_only=True))
print("END")