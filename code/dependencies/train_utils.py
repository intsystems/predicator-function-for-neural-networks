import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def load_json_from_directory(directory_path: Path) -> List[dict]:
    """
    Downloads all JSON files from a directory and its subdirectories.
    Adds the 'id' field from the file name, if missing.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    json_data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "id" not in data:
                        match = re.search(r"(\d+)\.json$", file)
                        if match:
                            data["id"] = int(match.group(1))
                    json_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
    return json_data


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    lr_final: float,
    max_epochs: int,
    warmup_epochs: int = 0,
):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=0.9,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    warmup_epochs = max(0, int(warmup_epochs))
    t_max = max(1, max_epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_final)

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    return optimizer, scheduler


def save_training_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint_payload, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: Path,
    map_location: str = "cpu",
) -> Optional[Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location=map_location)
