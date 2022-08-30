import time
from kanpsack import *

import numpy as np

a = np.array([
    [90, 100, 115, 120, 131, 143, 158, 168, 179, 185, 194, 205, 219, 226, 230, 243, 251, 263],
    [111, 125, 134, 140, 152, 169, 172, 182, 190, 202, 214, 224, 238, 248, 255, 268, 277, 280],
    [70, 89, 92, 102, 110, 123, 132, 140, 150, 160, 172, 181, 191, 202, 214, 226, 237, 243],
    [30, 45, 50, 66, 76, 82, 90, 106, 119, 121, 137, 143, 150, 165, 179, 187, 192, 208],
    [50, 69, 73, 85, 91, 105, 116, 128, 133, 145, 152, 164, 176, 183, 199, 205, 217, 229],
]
)


# a = np.array([[30 + 20 * i + j * 10 + np.random.randint(0, 10) for j in range(200)] for i in range(5)])


def is_feasible(sol, budget):
    return sum(sol) < budget


def value(sol, matrix):
    return sum([matrix[i][sol[i]] for i in range(len(matrix))])


def mkcp_solver(matrix):
    matrix = np.array(matrix)
    res_matrix = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix) + 1)]
    res_matrix[1] = matrix[0]
    sol_matrix = [[{'i_mat': 0, 'i_aux': 0} for i in range(len(matrix[0]))] for j in range(len(matrix))]
    sol_matrix[0] = [{'i_mat': 0, 'i_aux': i} for i in range(len(matrix[0]))]
    for i in range(2, len(matrix) + 1):
        for j in range(len(matrix[0])):
            elems = [matrix[i - 1][j - k] + res_matrix[i - 1][k] for k in range(j + 1)]
            best = elems.index(max(elems))
            sol_matrix[i - 1][j]['i_mat'] = best
            sol_matrix[i - 1][j]['i_aux'] = j - best
            res_matrix[i][j] = elems[best]
    sol = []

    k = np.argmax(res_matrix[len(matrix) - 1])
    for i in range(len(matrix) - 1, -1, -1):
        sol.append(sol_matrix[i][k]['i_aux'])
        k = sol_matrix[i][k]['i_mat']

    sol.reverse()
    return sol


start = time.time()
sol = mkcp_solver(a)
print(f'mkcp({sol, value(sol, a)}) in {time.time() - start}')
start = time.time()
sol = knapsack_optimizer(a)
print(f'knapsack({sol, value(sol, a)}) in {time.time() - start}')
value_sol = value(sol, a)
