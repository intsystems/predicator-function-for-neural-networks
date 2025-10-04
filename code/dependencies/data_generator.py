import random
import copy

from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import numpy as np
import multiprocessing as mp

DARTS_OPS = [
    # 'none',
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


def generate_cells(num_nodes, name="normal", operations=DARTS_OPS, rng=None):
    cells = dict()
    
    for i in range(num_nodes - 1):
        cur_indexes = list(range(0, i + 2))

        if rng is None:
            random_op_0, random_op_1 = random.choices(operations, k=2)
        elif hasattr(rng, 'choices'):
            random_op_0, random_op_1 = rng.choices(operations, k=2)
        else:
            random_op_0 = operations[int(rng.integers(0, len(operations)))]
            random_op_1 = operations[int(rng.integers(0, len(operations)))]

        if rng is None:
            random_index_0, random_index_1 = random.sample(cur_indexes, k=2)
        elif hasattr(rng, 'sample'):
            random_index_0, random_index_1 = rng.sample(cur_indexes, k=2)
        elif hasattr(rng, 'choice'):
            indices = rng.choice(cur_indexes, size=2, replace=False, shuffle=True)
            random_index_0, random_index_1 = int(indices[0]), int(indices[1])
        else:
            raise ValueError(f"Unsupported rng type: {type(rng)}")

        op_str_0 = f"{name}/op_{i + 2}_0"
        op_str_1 = f"{name}/op_{i + 2}_1"
        input_str_0 = f"{name}/input_{i + 2}_0"
        input_str_1 = f"{name}/input_{i + 2}_1"

        cells[op_str_0] = random_op_0
        cells[input_str_0] = [int(random_index_0)]
        cells[op_str_1] = random_op_1
        cells[input_str_1] = [int(random_index_1)]

    return cells


def generate_single_architecture(seed=None):
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    
    rng = np.random.Generator(np.random.PCG64(seed))
    
    normal_cell = generate_cells(5, name="normal", rng=rng)
    reduction_cell = generate_cells(5, name="reduce", rng=rng)
    
    tmp_dict = {**normal_cell, **reduction_cell}
    return {"architecture": tmp_dict, "seed": int(seed)}

def generate_unique_seeds(N_MODELS, low=1, high=int(1e9)):
    seeds = set()
    rng = np.random.default_rng()
    while len(seeds) < N_MODELS:
        print(f"preparing random seeds, progress:{len(seeds)}/{N_MODELS}", end='\r')
        seeds.add(int(rng.integers(low, high)))
    return list(seeds)

def generate_arch_dicts(N_MODELS, use_tqdm=False, n_jobs=None, batch_size=100):
    if n_jobs is None:
        n_jobs = min(8, mp.cpu_count())

    seeds = generate_unique_seeds(N_MODELS)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    def safe_generate_batch(seed_batch):
        return [generate_single_architecture(seed=s) for s in seed_batch]

    batches = list(chunks(seeds, batch_size))
    iterable = tqdm(batches, desc="Generating batches", total=len(batches)) if use_tqdm else batches

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(safe_generate_batch)(batch) for batch in iterable
    )

    return [arch for batch in results for arch in batch]



def mutate_architectures(
    arch_list, n_mutations=1, operations=DARTS_OPS, n_out_models=None
):
    if n_out_models is not None:
        arch_list = [random.choice(arch_list) for _ in range(n_out_models)]

    mutated = []
    for arch_entry in arch_list:
        arch = copy.deepcopy(arch_entry["architecture"])
        op_keys = [k for k in arch.keys() if k.startswith(("normal/op_", "reduce/op_"))]
        n = min(n_mutations, len(op_keys))
        mutate_keys = random.sample(op_keys, k=n)

        for key in mutate_keys:
            current_op = arch[key]
            choices = [op for op in operations if op != current_op]
            arch[key] = random.choice(choices)
        mutated.append({"architecture": arch})

    return mutated


def load_dataset(config) -> None:
    config.dataset_path = Path(config.dataset_path)

    config.models_dict_path = []
    for file_path in tqdm(config.dataset_path.rglob("*.json"), desc="Loading dataset"):
        config.models_dict_path.append(file_path)


def load_dataset_on_inference(config) -> None:
    config.dataset_path = Path(config.prepared_dataset_path)

    config.models_dict_path = []
    for file_path in config.dataset_path.rglob("*.json"):
        config.models_dict_path.append(file_path)
