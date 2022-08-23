import numpy as np


def get_graph_paths(graph, node):
    aux = graph.copy()
    q = [{'matrix': aux, 'node': node, 'path': []}]
    paths = []
    while len(q) > 0:

        # removes the item from the matrix and saves parameters
        elem = q.pop(0)
        matrix = elem['matrix']
        node = elem['node']
        elem['path'].append(node)

        # adds to the total result the clicks of sub_nodes
        paths.append(elem['path'].copy())
        # avoids that next iterations will return to this node
        matrix[:, node] = 0
        if np.all((matrix == 0)):
            continue

        # adds the matrices to be explored for all children
        for i in [i for i, value in enumerate(matrix[node]) if value != 0]:
            q.append(
                {'matrix': matrix.copy(), 'node': i, 'path': elem['path'].copy()})
            # print(f''' path: {elem['path']}, {clicks * matrix[node][i]} clicks arrived to {i + 1} ''')
    return paths
