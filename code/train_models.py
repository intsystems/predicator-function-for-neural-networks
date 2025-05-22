import os
import json
import numpy as np
import torch
import nni
from torch.utils.data import SubsetRandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.nas.evaluator.pytorch import DataLoader, Classification
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.space import model_context
from tqdm import tqdm
from IPython.display import clear_output
from nni.nas.evaluator.pytorch import Lightning, Trainer

from dependecies.data_generator import generate_arch_dicts
from dependecies.darts_classification_module import DartsClassificationModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST = False


ARCHITECTURES_PATH = "/kaggle/input/second-dataset/dataset"
MAX_EPOCHS = 60
LEARNING_RATE = 0.025
BATCH_SIZE = 96
NUM_MODLES = 2000
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

SEED = 228
# random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # если есть GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_json_from_directory(directory_path):
    json_data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        json_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {file_path}: {e}")
    return json_data


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
    split = int(num_samples * 0.5)

    search_train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        sampler=SubsetRandomSampler(indices[:split]),
    )

    search_valid_loader = DataLoader(
            train_data,
        batch_size=batch_size,
        num_workers=10,
        sampler=SequentialSampler(indices[split:]),
    )

    return search_train_loader, search_valid_loader


def train_model(
    architecture, 
    train_loader, 
    valid_loader, 
    max_epochs=600, 
    learning_rate=0.025,
    fast_dev_run=False
):
    with model_context(architecture):
        model = DartsSpace(width=16, num_cells=10, dataset='cifar')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.to(device)

    evaluator = Lightning(
        DartsClassificationModule(
            learning_rate=learning_rate,
            weight_decay=3e-4,
            auxiliary_loss_weight=0.4,
            max_epochs=max_epochs
        ),
        trainer=Trainer(
            gradient_clip_val=5.0,
            max_epochs=max_epochs,
            fast_dev_run=fast_dev_run,
            devices=[0]
        ),
        train_dataloaders=train_loader#,
        #val_dataloaders=valid_loader
    )

    evaluator.fit(model)
    return model


def evaluate_and_save_results(
    model,
    architecture,
    model_id,  # Новый обязательный параметр для идентификации модели
    valid_loader,
    folder_name="results_seq_0"
):
    """
    Оценивает модель на валидационном наборе данных и сохраняет результаты в JSON.
    Аргументы:
    model: Обученная модель
    architecture: Архитектура модели
    valid_loader (DataLoader): DataLoader для валидационных данных
    model_id: Уникальный идентификатор модели
    folder_name (str): Папка для сохранения результатов
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(folder_name, exist_ok=True)

    # Перенос модели на устройство и режим оценки
    model.to(device)
    model.eval()

    valid_correct = 0
    valid_total = 0
    valid_preds = []

    with torch.no_grad():
        for images, labels in valid_loader:
            # print(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            valid_preds.extend(outputs.cpu().tolist())
            _, predicted = torch.max(outputs, 1)
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)

    valid_accuracy = valid_correct / valid_total

    # Формирование результата
    result = {
        "architecture": architecture,
        "valid_predictions": valid_preds,
        "valid_accuracy": valid_accuracy,
    }

    # Генерация имени файла с использованием model_id
    file_name = f"model_{model_id}_results.json"
    file_path = os.path.join(folder_name, file_name)

    # Сохранение результатов
    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results for model_{model_id} saved to {file_path}")


if __name__ == "__main__":
    arch_dicts = generate_arch_dicts(NUM_MODLES)
    arch_dicts = [tmp_arch["architecture"] for tmp_arch in arch_dicts]
    search_train_loader, search_valid_loader = get_data_loaders(
        batch_size=BATCH_SIZE
    )  # Получаем загрузчики CIFAR10

    for idx, architecture in enumerate(tqdm(arch_dicts)):
        model = train_model(  # Обучаем модель
            architecture,
            search_train_loader,
            search_valid_loader,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            fast_dev_run=False
        )
        clear_output(wait=True)
        
        evaluate_and_save_results(
            model, architecture, idx, valid_loader=search_valid_loader, folder_name="results_seq_0"
        )  # Оцениваем и сохраняем архитектуры, предсказания на тестовом наборе CIFAR10 и accuracy
