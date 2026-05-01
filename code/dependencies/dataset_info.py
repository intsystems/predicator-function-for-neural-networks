from typing import Dict, Any

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST


class Cutout:
    """Apply cutout to an image tensor."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be Tensor. Got {type(img)}")

        device = img.device
        c, h, w = img.size()
        mask = torch.ones((h, w), dtype=torch.float32, device=device)

        y = torch.randint(0, h, (1,), device=device).item()
        x = torch.randint(0, w, (1,), device=device).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask[y1:y2, x1:x2] = 0.0
        mask = mask.unsqueeze(0).expand_as(img)
        return img * mask


def duplicate_channel(x: torch.Tensor) -> torch.Tensor:
    """Duplicates a single-channel image into 3 channels."""
    return x.repeat(3, 1, 1)


class DatasetsInfo:
    """Configuration of datasets: transformations, normalization, classes."""

    DATASETS = {
        "cifar10": {
            "key": "cifar",
            "class": CIFAR10,
            "num_classes": 10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                    Cutout(16),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                    ),
                ]
            ),
        },
        "cifar100": {
            "key": "cifar100",
            "class": CIFAR100,
            "num_classes": 100,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2673, 0.2564, 0.2762],
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]
                    ),
                    Cutout(16),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]
                    ),
                ]
            ),
        },
        "fashionmnist": {
            "key": "cifar",
            "class": FashionMNIST,
            "num_classes": 10,
            "mean": [0.2860406] * 3,
            "std": [0.35302424] * 3,
            "img_size": 32,
            "train_transform": transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Lambda(duplicate_channel),
                    transforms.Normalize(mean=[0.2860406] * 3, std=[0.35302424] * 3),
                    Cutout(12),
                ]
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(duplicate_channel),
                    transforms.Normalize(mean=[0.2860406] * 3, std=[0.35302424] * 3),
                ]
            ),
        },
    }

    @classmethod
    def get(cls, dataset_name: str) -> Dict[str, Any]:
        """Returns the configuration of a dataset."""
        dataset_name = dataset_name.lower()
        if dataset_name not in cls.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}"
            )
        return cls.DATASETS[dataset_name]
