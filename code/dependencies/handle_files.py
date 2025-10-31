from pathlib import Path
import shutil
import json
import re
import numpy as np
from typing import Optional, List, Tuple, Dict


def collect_models(src_dir: str):
    """
    Собирает архитектуры и веса моделей из выходных директорий.

    Копирует пары .json и .pth файлов с одинаковыми ID в отдельные директории.

    Args:
        src_dir (str): Путь к исходной директории.
    """
    src_path = Path(src_dir)

    json_dir = src_path / "collected_json"
    pth_dir = src_path / "collected_pth"
    json_dir.mkdir(parents=True, exist_ok=True)
    pth_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    for greed_folder in sorted(src_path.glob("output_greed_*")):
        if not greed_folder.is_dir():
            continue

        arch_dir = greed_folder / "trained_models_archs"
        weight_dir = greed_folder / "trained_models"

        if not arch_dir.exists() or not weight_dir.exists():
            print(f"⚠ Пропущено {greed_folder}: нет нужных директорий")
            continue

        json_files = sorted(arch_dir.glob("*.json"))
        pth_files = sorted(weight_dir.glob("*.pth"))

        # функция для извлечения числового id из имени файла
        def extract_id(path: Path):
            m = re.search(r"\d+", path.stem)
            return int(m.group()) if m else None

        # создаём словари id → Path
        json_map = {extract_id(f): f for f in json_files if extract_id(f) is not None}
        pth_map = {extract_id(f): f for f in pth_files if extract_id(f) is not None}

        # пересечение по id
        common_ids = sorted(json_map.keys() & pth_map.keys())

        for file_id in common_ids:
            shutil.copy2(json_map[file_id], json_dir / f"model_{idx}.json")
            shutil.copy2(pth_map[file_id], pth_dir / f"model_{idx}.pth")
            idx += 1

    print(f"Собрано {idx-1} пар моделей. JSON → {json_dir}, PTH → {pth_dir}")


def process_json_logits(src_dir: str, dst_dir: str):
    """
    Обрабатывает JSON файлы, заменяя логиты на предсказания.

    Загружает JSON файлы из исходной директории, извлекает массив логитов
    из поля 'valid_predictions', находит индекс класса с максимальным значением
    для каждого примера и заменяет исходный массив логитов на массив предсказаний.
    Результат сохраняется в новой директории.

    Args:
        src_dir (str): Путь к исходной директории с JSON файлами.
        dst_dir (str): Путь к директории для сохранения обработанных файлов.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    for json_file in src_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "valid_predictions" not in data:
            print(f"⚠ Пропущен {json_file} — нет поля 'valid_predictions'")
            continue

        logits = np.array(data["valid_predictions"])
        predictions = logits.argmax(axis=1).tolist()
        data["valid_predictions"] = predictions  # заменяем логиты на предсказания

        dst_file = dst_path / json_file.name
        with open(dst_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Готово. Новые json сохранены в {dst_path}")


ID_RE = re.compile(r"(\d+)")


def extract_id(name: str) -> Optional[int]:
    """Возвращает первую найденную последовательность цифр из имени файла."""
    m = ID_RE.search(name)
    return int(m.group(1)) if m else None


def unique_path(path: Path) -> Path:
    """Возвращает уникальный путь, добавляя _1, _2... если файл уже существует."""
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{base}_{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1


def collect_files(dir_path: Path, exts) -> Dict[int, Path]:
    """Собирает файлы с нужным расширением и индексом."""
    mapping = {}
    if dir_path.exists():
        for p in dir_path.iterdir():
            if p.suffix.lower() in exts:
                if (id_ := extract_id(p.name)) is not None:
                    mapping[id_] = p
    return mapping


def collect_and_match(
    pairs: List[Tuple[str, str]],
    dst_root: str,
    arch_exts: Tuple[str, ...] = (".json", ".yaml", ".yml"),
    weight_exts: Tuple[str, ...] = (".pth", ".pt", ".ckpt", ".h5"),
    move: bool = False,
    dry_run: bool = False,
):
    """
    Собирает и сопоставляет файлы архитектур и весов моделей.

    Проходит по парам директорий, содержащих архитектуры и веса моделей соответственно.
    Извлекает ID из имен файлов и сопоставляет файлы по этим ID.
    Сопоставленные пары копируются или перемещаются в целевые директории.

    Args:
        pairs (List[Tuple[str, str]]): Список пар путей (архитектура, веса).
        dst_root (str): Корневая директория для сохранения собранных файлов.
        arch_exts (Tuple[str, ...]): Расширения файлов архитектур. По умолчанию (".json", ".yaml", ".yml").
        weight_exts (Tuple[str, ...]): Расширения файлов весов. По умолчанию (".pth", ".pt", ".ckpt", ".h5").
        move (bool): Если True, файлы перемещаются, а не копируются. По умолчанию False.
        dry_run (bool): Если True, выполняется пробный прогон без реальных операций. По умолчанию False.
    """
    dst_arch, dst_weights = Path(dst_root) / "architectures", Path(dst_root) / "weights"
    dst_arch.mkdir(parents=True, exist_ok=True)
    dst_weights.mkdir(parents=True, exist_ok=True)

    action = shutil.move if move else shutil.copy2

    global_index = 0
    for pair_index, (arch_dir, weight_dir) in enumerate(pairs, start=1):
        arch_map = collect_files(Path(arch_dir), arch_exts)
        weight_map = collect_files(Path(weight_dir), weight_exts)

        matched_ids = sorted(set(arch_map) & set(weight_map))

        for local_idx, id_ in enumerate(matched_ids, start=1):
            if pair_index == 1:
                # Первая пара — сохраняем оригинальные индексы
                new_id = id_
            else:
                # Все следующие пары — перенумеровываем
                new_id = global_index + local_idx

            arch_dest = unique_path(dst_arch / f"model_{new_id}{arch_map[id_].suffix}")
            weight_dest = unique_path(
                dst_weights / f"model_{new_id}{weight_map[id_].suffix}"
            )

            print(f"[{id_} -> {new_id}] {arch_map[id_]} -> {arch_dest}")
            print(f"[{id_} -> {new_id}] {weight_map[id_]} -> {weight_dest}")

            if not dry_run:
                action(arch_map[id_], arch_dest)
                action(weight_map[id_], weight_dest)

        # Обновляем глобальный счётчик моделей после каждой пары
        global_index += len(matched_ids)


# collect_models("datasets/tmp")
# process_json_logits("datasets/tmp/collected_json", "datasets/tmp/collected_json_max")

# collect_and_match(
#     [
#         (
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part1/trained_models_archs_1",
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part1/trained_models_pth_1",
#         ),
#         (
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part2/trained_models_archs_1",
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part2/trained_models_pth_1",
#         ),
#         (
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part3/trained_models_archs_1",
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part3/trained_models_pth_1",
#         ),
#         (
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part4/trained_models_archs_1",
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part4/trained_models_pth_1",
#         ),
#         (
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part5/trained_models_archs_1",
#             "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div_part5/trained_models_pth_1",
#         ),
#     ],
#     "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div",
# )


collect_and_match(
    [
        (
            "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div/architectures",
            "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_div/weights",
        ),
        (
            "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_acc_archs/trained_models_archs_1",
            "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_acc_archs/trained_models_pth_1",
        ),
    ],
    "/home/udeneev-av/RAS/predicator-function-for-neural-networks/code/datasets/CIFAR10_united",
)
