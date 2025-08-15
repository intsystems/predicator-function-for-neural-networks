from pathlib import Path
import shutil
import json
import re
import numpy as np
from typing import Optional, List, Tuple, Dict

def collect_models(src_dir: str):
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


def _extract_id(name: str) -> Optional[str]:
    """Return first numeric token found in filename or None."""
    m = ID_RE.search(name)
    return m.group(1) if m else None


def _unique_dest(dest: Path) -> Path:
    """If dest exists, append suffix _1, _2, ... to avoid overwrite."""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def collect_and_match(
    pairs: List[Tuple[str, str]],
    dst_root: str,
    arch_exts: Tuple[str, ...] = (".json", ".yaml", ".yml"),
    weight_exts: Tuple[str, ...] = (".pth", ".pt", ".ckpt", ".h5"),
    move: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Собирает согласованные пары (архитектуры, веса) из нескольких пар директорий.

    Параметры
    ---------
    pairs: list of (arch_dir, weights_dir)
        Массив пар путей в файловой системе (строки). Внутри пары файлы согласованы по
        числовому идентификатору, который извлекается как первая встреченная цифросерия
        в имени файла (например, "model_123.json" -> id="123").

    dst_root: путь к корневой директории, куда поместятся две папки:
        dst_root/architectures  и dst_root/weights

    arch_exts, weight_exts: допустимые расширения для файлов архитектур и весов.

    move: если True - перемещать файлы (shutil.move), иначе копировать (shutil.copy2).
    dry_run: если True - не выполнять копирование/перемещение, только симулировать.

    Возвращает словарь-отчёт с числом обработанных и найденных несопоставленных файлов.
    """
    dst_root = Path(dst_root)
    dst_arch = dst_root / "architectures"
    dst_weights = dst_root / "weights"
    dst_arch.mkdir(parents=True, exist_ok=True)
    dst_weights.mkdir(parents=True, exist_ok=True)

    arch_map: Dict[str, List[Path]] = {}
    weight_map: Dict[str, List[Path]] = {}

    for arch_dir, weight_dir in pairs:
        arch_dir_p = Path(arch_dir)
        weight_dir_p = Path(weight_dir)

        if arch_dir_p.exists():
            for p in arch_dir_p.iterdir():
                if p.is_file() and p.suffix.lower() in arch_exts:
                    id_ = _extract_id(p.name)
                    if id_:
                        arch_map.setdefault(id_, []).append(p)
        if weight_dir_p.exists():
            for p in weight_dir_p.iterdir():
                if p.is_file() and p.suffix.lower() in weight_exts:
                    id_ = _extract_id(p.name)
                    if id_:
                        weight_map.setdefault(id_, []).append(p)

    matched = 0
    skipped_multiple = 0
    unmatched_arch = []
    unmatched_weights = []

    all_ids = set(arch_map.keys()) | set(weight_map.keys())
    for id_ in sorted(all_ids, key=lambda x: int(x)):
        arch_list = arch_map.get(id_, [])
        weight_list = weight_map.get(id_, [])

        if arch_list and weight_list:
            # take first occurrence from each side (warn if multiples)
            if len(arch_list) > 1 or len(weight_list) > 1:
                skipped_multiple += 1
            arch_file = arch_list[0]
            weight_file = weight_list[0]

            dest_arch = _unique_dest(dst_arch / arch_file.name)
            dest_weight = _unique_dest(dst_weights / weight_file.name)

            action = "move" if move else "copy"
            print(f"[{id_}] {action}: {arch_file} -> {dest_arch}; {weight_file} -> {dest_weight}")

            if not dry_run:
                if move:
                    shutil.move(str(arch_file), str(dest_arch))
                    shutil.move(str(weight_file), str(dest_weight))
                else:
                    shutil.copy2(str(arch_file), str(dest_arch))
                    shutil.copy2(str(weight_file), str(dest_weight))
            matched += 1
        else:
            if arch_list and not weight_list:
                unmatched_arch.append((id_, [p.name for p in arch_list]))
            if weight_list and not arch_list:
                unmatched_weights.append((id_, [p.name for p in weight_list]))

    report = {
        "matched_ids": matched,
        "ids_with_multiple_files": skipped_multiple,
        "unmatched_architectures": len(unmatched_arch),
        "unmatched_weights": len(unmatched_weights),
    }

    print("\nReport:")
    print(report)
    if unmatched_arch:
        print("Unmatched architectures (id -> files):")
        for id_, files in unmatched_arch:
            print(f"  {id_} -> {files}")
    if unmatched_weights:
        print("Unmatched weights (id -> files):")
        for id_, files in unmatched_weights:
            print(f"  {id_} -> {files}")

    return report


# collect_models("datasets/tmp")
# process_json_logits("datasets/tmp/collected_json", "datasets/tmp/collected_json_max")

collect_and_match(
    [("datasets/tmp/collected_json_1", "datasets/tmp/collected_pth_1"), ("datasets/tmp/collected_json_2", "datasets/tmp/collected_pth_2")],
    "datasets/tmp/collected_json_max",
)
