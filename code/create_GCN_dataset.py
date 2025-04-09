import pandas as pd
import numpy as np
import os
from scipy.special import softmax
import json
from tqdm import tqdm 

def load_json_from_directory(directory_path):
    json_data = []
    for root, _, files in (os.walk(directory_path)):
        for file in tqdm(files, desc="Processing JSON files"):
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        json_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {file_path}: {e}")
    return json_data

def apply_softmax_to_predictions(data):

    for item in tqdm(data, desc="Applying softmax to predictions"):
        if "test_predictions" in item:
            predictions = np.array(item["test_predictions"])
            softmaxed = softmax(predictions, axis=1)
            item["test_predictions"] = softmaxed.tolist()

def save_dicts_as_json(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, item in enumerate(tqdm(data, desc="Saving JSON files")):
        file_name = f"sample_{i:04d}.json"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=4)

def apply_argmax_to_predictions(data):
    for item in tqdm(data, desc="Applying argmax to predictions"):
        if "test_predictions" in item:
            predictions = np.array(item["test_predictions"])
            argmaxed = np.argmax(predictions, axis=1)
            item["test_predictions"] = argmaxed.tolist()


dir_path = "dataset_logits_fixed"
first_arch_dicts = load_json_from_directory(dir_path)

output_dir = "dataset_probs"
apply_softmax_to_predictions(first_arch_dicts)
save_dicts_as_json(first_arch_dicts, output_dir)

apply_argmax_to_predictions(first_arch_dicts)
output_dir = "tmp_dataset"
save_dicts_as_json(first_arch_dicts, output_dir)

second_arch_dicts = load_json_from_directory("second_dataset")

first_arch_dicts.extend(second_arch_dicts)
output_dir = "third_dataset"
save_dicts_as_json(first_arch_dicts, output_dir)