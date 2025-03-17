import os
import json
import numpy as np
import torch
import nni
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.nas.evaluator.pytorch import DataLoader, Classification
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.space import model_context
from tqdm import tqdm
from IPython.display import clear_output

ARCHITECTURES_PATH = "dataset/arch_dicts.json"
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


def load_arch_dicts(json_path):
    """
    Загружает словари архитектур из JSON файла.

    Аргументы:
        json_path (str): Путь к JSON файлу, содержащему словари архитектур.

    Возвращает:
        dict: Словарь, содержащий конфигурации архитектур.
    """
    with open(json_path, "r") as f:
        arch_dicts = json.load(f)
    return arch_dicts


def get_data_loaders(batch_size=512):
    """
    Возвращает загрузчики данных для обучения и валидации.

    Параметры:
    batch_size (int): Размер батча для загрузчиков данных. По умолчанию 1024.

    Возвращает:
    tuple: Кортеж, содержащий два объекта DataLoader:
        - search_train_loader: Загрузчик данных для обучения.
        - search_valid_loader: Загрузчик данных для валидации.
    """
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_data = nni.trace(CIFAR10)(
        root="./data", train=True, download=True, transform=transform
    )
    num_samples = len(train_data)
    indices = np.random.permutation(num_samples)
    split = num_samples // 2

    search_train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=6,
        sampler=SubsetRandomSampler(indices[:split]),
    )

    search_valid_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=6,
        sampler=SubsetRandomSampler(indices[split:]),
    )

    return search_train_loader, search_valid_loader


def train_model(
    architecture, train_loader, valid_loader, max_epochs=10, learning_rate=1e-3
):
    """
    Обучает модель на основе заданной архитектуры и данных.
    Параметры:
    architecture (str): Архитектура модели, которая будет использоваться.
    train_loader (DataLoader): DataLoader для обучающих данных.
    valid_loader (DataLoader): DataLoader для валидационных данных.
    max_epochs (int, необязательно): Максимальное количество эпох для обучения. По умолчанию 10.
    learning_rate (float, необязательно): Скорость обучения. По умолчанию 1e-3.
    Возвращает:
    model: Обученная модель.
    """
    with model_context(architecture):
        model = DartsSpace(width=16, num_cells=3, dataset="cifar")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # Enable multi-GPU training

    model.to(device)

    evaluator = Classification(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=max_epochs,
        num_classes=10,
        export_onnx=False,  # Disable ONNX export for this experiment
        fast_dev_run=False,  # Should be false for fully training
    )

    evaluator.fit(model)
    return model

def evaluate_and_save_results(
    models, architectures, batch_size=512, num_workers=6, folder_name="results"
):
    """
    Оценивает модели на тестовом наборе данных CIFAR-10 и сохраняет результаты в файлы JSON.
    Аргументы:
    models (list): Список обученных моделей.
    architectures (list): Список архитектур моделей.
    batch_size (int, необязательно): Размер батча для загрузчика данных. По умолчанию 1024.
    num_workers (int, необязательно): Количество потоков для загрузчика данных. По умолчанию 6.
    folder_name (str, необязательно): Имя папки для сохранения результатов. По умолчанию "results".
    Исключения:
    ValueError: Если количество моделей и архитектур не совпадает.
    Результаты:
    Для каждой модели создается файл JSON с результатами, содержащий:
    - architecture: Архитектура модели.
    - test_predictions: Предсказания модели на тестовом наборе данных.
    - test_accuracy: Точность модели на тестовом наборе данных.
    """
    if len(models) != len(architectures):
        raise ValueError("Количество моделей и архитектур должно совпадать")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(folder_name, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    for i, (model, architecture) in enumerate(zip(models, architectures)):
        model.to(device)
        model.eval()

        test_correct = 0
        test_total = 0
        test_preds = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().tolist())
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total

        result = {
            "architecture": architecture,
            "test_predictions": test_preds,
            "test_accuracy": test_accuracy,
        }

        file_name = f"model_{i+1}_results.json"
        file_path = os.path.join(folder_name, file_name)

        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Results for model_{i + 1} saved to {file_path}")


if __name__ == "__main__":
    arch_dicts = load_arch_dicts(ARCHITECTURES_PATH)  # Загружаем словари архитектур
    search_train_loader, search_valid_loader = get_data_loaders(
        batch_size=BATCH_SIZE
    )  # Получаем загрузчики CIFAR10

    models = []
    architectures = []
    for architecture in tqdm(arch_dicts):
        model = train_model(  # Обучаем модель
            architecture,
            search_train_loader,
            search_valid_loader,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        models.append(model)
        architectures.append(architecture)
        clear_output(wait=True)

    evaluate_and_save_results(
        models, architectures, batch_size=BATCH_SIZE
    )  # Оцениваем и сохраняем архитектуры, предсказания на тестовом наборе CIFAR10 и accuracy
