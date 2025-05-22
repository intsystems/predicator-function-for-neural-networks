import numpy as np
import re
import torch
from sklearn.preprocessing import OneHotEncoder
from graphviz import Digraph
from IPython.display import display

DARTS_OPS = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ]

encoder = OneHotEncoder(handle_unknown='ignore')
encoder = OneHotEncoder(handle_unknown='ignore')
ops_array = np.array(DARTS_OPS).reshape(-1, 1)

DARTS_OPS_ONE_HOT = encoder.fit_transform(ops_array).toarray()

def extract_cells(arch_dict):
    normal_cell, reduction_cell = [], []
    tmp_list = []

    for key, value in arch_dict["architecture"].items():
        if key.startswith("normal/") or key.startswith("reduce/"):
            tmp_list.extend([key, value])

        if len(tmp_list) == 4:
            tmp_list.pop(2)
            if key.startswith("normal/"):
                normal_cell.append(tmp_list)
            else:
                reduction_cell.append(tmp_list)
            tmp_list = []

    return normal_cell, reduction_cell

class Vertex:
    def __init__(self, op, in_channel, out_channel):
        self.op = op
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.op_one_hot = DARTS_OPS_ONE_HOT[DARTS_OPS.index(op)]

    def __str__(self):
        return f"Op: {self.op} | In: {self.in_channel} | Out: {self.out_channel}"
    def __repr__(self):
        return self.__str__()
    
class Graph(torch.utils.data.Dataset):
    def __init__(self, model_dict, index=0):
        self.model_dict = model_dict
        self.normal_cell, self.reduction_cell = extract_cells(model_dict)

        self._normal_graph = self.make_graph(self.normal_cell)
        self._reduction_graph = self.make_graph(self.reduction_cell)

        self.normal_num_vertices, self.reduction_num_vertices = self.__len__()

        self.graph = self.make_full_graph()
        self.index = index

    def __len__(self):
        max_normal_out = max(vertex.out_channel for vertex in self._normal_graph)
        max_reduction_out = max(vertex.out_channel for vertex in self._reduction_graph)
        return max_normal_out, max_reduction_out

    def graph_size(self, graph):
        return max((vertex.out_channel for vertex in graph), default=0)

    def make_full_graph(self):
        graph = [vertex for vertex in self._normal_graph]
        graph = self._unite_graphs(graph, self._reduction_graph)

        max_channel_diff, _ = self.__len__()
        graph.append(Vertex("none", max_channel_diff * 2 + 1, max_channel_diff * 2 + 1))

        return graph

    def _unite_graphs(self, graph1, graph2):
        graph1_size = self.graph_size(graph1)
        new_graph = [vertex for vertex in graph1]
        for vertex in graph2:
            new_vertex = Vertex(
                vertex.op,
                vertex.in_channel + graph1_size,
                vertex.out_channel + graph1_size,
            )
            new_graph.append(new_vertex)

        new_graph.sort(key=lambda vertex: (vertex.in_channel, vertex.out_channel))

        return new_graph

    def make_graph(self, cell):
        graph = []
        for value in cell:
            in_channel = int(value[2][0])
            out_channel = int(re.search(r"op_(\d+)_", value[0]).group(1))
            op = value[1]
            graph.append(Vertex(op, in_channel, out_channel))
        graph.append(Vertex("none", 0, 0))
        graph.append(Vertex("none", 1, 1))

        graph.sort(key=lambda vertex: (vertex.in_channel, vertex.out_channel))

        return graph

    def show_graph(self):
        adj_matrix, operations, _ = self.get_adjacency_matrix()
        graph_name = "Graph"

        dot = Digraph(comment=graph_name, format="png")
        dot.attr(rankdir="TB")

        num_nodes = len(self.graph)

        # Добавляем узлы с оригинальными метками
        for idx, vertex in enumerate(self.graph):
            label = (
                f"{{Op: {vertex.op} | "
                f"In: {vertex.in_channel} | "
                f"Out: {vertex.out_channel}}}"
            )
            dot.node(str(idx), label=label, shape="record")

        # Добавляем связи на основе матрицы смежности
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] == 1:
                    dot.edge(str(i), str(j))

        display(dot)

    def get_normal_graph(self):
        return self._normal_graph

    def get_reduction_graph(self):
        return self._reduction_graph

    def get_adjacency_matrix(self):
        adj_matrix_size = len(self.graph)
        max_channel_diff, _ = self.__len__()
        adj_matrix = np.zeros(shape=(adj_matrix_size, adj_matrix_size))

        operations = [vertex.op for vertex in self.graph]
        operations_one_hot = [vertex.op_one_hot for vertex in self.graph]
        for i in range(adj_matrix_size):
            for j in range(adj_matrix_size):
                if j == i:
                    continue
                vertex_1 = self.graph[i]
                vertex_2 = self.graph[j]

                if (vertex_1.out_channel == vertex_2.in_channel) and (
                    (
                        vertex_1.in_channel <= max_channel_diff
                        and vertex_2.out_channel <= max_channel_diff
                    )
                    or (
                        vertex_1.in_channel >= max_channel_diff
                        and vertex_2.out_channel >= max_channel_diff
                    )
                ):

                    adj_matrix[i, j] = 1

                if (  # Добавляем ребро из c_k на вход следующей клетке
                    (vertex_1.op == "none")
                    and (vertex_2.op == "none")
                    and (vertex_1.out_channel == 1)
                    and (vertex_2.in_channel == 6)
                ):
                    adj_matrix[i, j] = 1

        # Соединим оставшиеся узлы с выходом.

        for i in range(adj_matrix_size):
            for j in range(adj_matrix_size):
                if j == i:
                    continue
                vertex_1 = self.graph[i]
                vertex_2 = self.graph[j]

                if (np.all(adj_matrix[i, :] == 0)) and (
                    (
                        (vertex_2.op == "none")
                        and (vertex_2.in_channel == max_channel_diff)
                        and (vertex_1.in_channel < max_channel_diff)
                    )
                    or (
                        (vertex_2.out_channel == 2 * max_channel_diff + 1)
                        and (vertex_1.out_channel > max_channel_diff)
                    )
                ):
                    adj_matrix[i, j] = 1

        adj_matrix = np.array(adj_matrix)
        operations_one_hot = np.array(operations_one_hot)
        return adj_matrix, operations, operations_one_hot