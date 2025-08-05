import torch
from torch import nn
import torch.nn.functional as F

from GCN import GAT


class SiameseGAT(nn.Module):
    def __init__(self, input_dim, emb_dim=64, dropout=0.5, pooling="max", heads=4, similarity="l2"):
        super().__init__()
        self.encoder = GAT(input_dim, emb_dim, dropout, pooling, heads)
        self.similarity = similarity

    def forward(self, data1, data2):
        # data1, data2 — это батчи графов PyG Data
        e1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        e2 = self.encoder(data2.x, data2.edge_index, data2.batch)

        if self.similarity == "cosine":
            return F.cosine_similarity(e1, e2)
        elif self.similarity == "l2":
            return torch.linalg.vector_norm(e1 - e2, ord=2, dim=-1)
        else:
            raise ValueError("Unsupported similarity metric")
