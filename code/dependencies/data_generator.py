import random
import nni
import numpy as np

import io
import graphviz
import matplotlib.pyplot as plt
from PIL import Image

import json
import os

from tqdm.notebook import tqdm
from joblib import Parallel, delayed


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


def generate_cells(num_nodes, name='normal', operations=DARTS_OPS):
    cells = dict()
    for i in range(num_nodes - 1):
        cur_indexes = list(range(0, i + 2))
        random_op_0, random_op_1 = random.choices(operations, k=2)
        
        random_index_0, random_index_1 = random.sample(cur_indexes, k=2)
        
        op_str_0 = f'{name}/op_{i + 2}_0'
        op_str_1 = f'{name}/op_{i + 2}_1'
        input_str_0 = f'{name}/input_{i + 2}_0'
        input_str_1 = f'{name}/input_{i + 2}_1'
        
        cells[op_str_0] = random_op_0
        cells[input_str_0] = [random_index_0]
        cells[op_str_1] = random_op_1
        cells[input_str_1] = [random_index_1]
        
    return cells

def generate_single_architecture():
    normal_cell = generate_cells(5, name='normal')
    reduction_cell = generate_cells(5, name='reduce')
    tmp_dict = {**normal_cell, **reduction_cell}
    return {'architecture': tmp_dict}

def generate_arch_dicts(N_MODELS, use_tqdm=False):
    iterable = range(N_MODELS)
    if use_tqdm:
        iterable = tqdm(iterable)

    arch_dicts = Parallel(n_jobs=-1)(delayed(generate_single_architecture)() for _ in iterable)
    return arch_dicts