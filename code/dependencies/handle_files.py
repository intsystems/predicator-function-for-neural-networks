from pathlib import Path
import shutil
import json
import numpy as np

def collect_models(src_dir: str):
    src_path = Path(src_dir)

    json_dir = src_path / "collected_json"
    pth_dir = src_path / "collected_pth"
    json_dir.mkdir(parents=True, exist_ok=True)
    pth_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    for greed_folder in sorted(src_path.glob("output_greed_*")):
        if greed_folder.is_dir():
            arch_dir = greed_folder / "trained_models_archs"
            weight_dir = greed_folder / "trained_models"

            if not arch_dir.exists() or not weight_dir.exists():
                continue

            json_files = sorted(arch_dir.glob("*.json"))
            pth_files = sorted(weight_dir.glob("*.pth"))

            # предполагаем, что индексы внутри папки согласованы и одинаковое количество файлов
            for jf, pf in zip(json_files, pth_files):
                shutil.copy2(jf, json_dir / f"model_{idx}.json")
                shutil.copy2(pf, pth_dir / f"model_{idx}.pth")
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


# process_json_logits("../datasets/cifar100_json", "../datasets/cifar100_json_max")
collect_models("datasets/tmp")
