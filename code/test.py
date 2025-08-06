from pathlib import Path
import shutil
import re

src_dir = Path("datasets/third_dataset")
dst_dir = Path("datasets/third_dataset_2")

old_prefix = "sample_"
new_prefix = "model_"

dst_dir.mkdir(parents=True, exist_ok=True)

pattern = re.compile(rf"^{re.escape(old_prefix)}0*(\d+)\.json$")

for file_path in src_dir.glob(f"{old_prefix}*.json"):
    match = pattern.match(file_path.name)
    if not match:
        continue  # пропускаем, если имя не соответствует формату
    
    number = int(match.group(1))  # убираем ведущие нули автоматически
    new_name = f"{new_prefix}{number}.json"
    new_path = dst_dir / new_name
    shutil.copy(file_path, new_path)

print("✅ Готово!")
