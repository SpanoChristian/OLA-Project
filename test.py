import numpy as np
from graphviz import Digraph
from graphviz import Source
from config import *
graph_spec = ''
adj_matrix = np.array(config['graph']['adj_matrix'])
products = np.array(config['products'])
for i, product in enumerate(products):
    connected = "".join([product + '->' + item + '\n' for item in [products[j] + f' [label={value}]' for j, value in enumerate(adj_matrix[i]) if value != 0]])
    graph_spec += connected
graph_spec = 'digraph{' + graph_spec + '}'
dot = Source(graph_spec)
dot



