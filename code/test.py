import torch

a = 1
b = torch.tensor([1, 1, 1])
if torch.all(a == b):       # вернёт 0‑d tensor с True/False
    print("hello world")