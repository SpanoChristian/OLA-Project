import time

import numpy as np
# from knapsack import *

a = [
    [90, 100, 115, 120, 131, 143, 158, 168, 179, 185, 194],
    [111, 125, 134, 140, 152, 169, 172, 182, 190, 202, 214],
    [70, 89, 92, 102, 110, 123, 132, 140, 150, 160, 172],
    [30, 45, 50, 66, 76, 82, 90, 106, 119, 121, 137],
    [50, 69, 73, 85, 91, 105, 116, 128, 133, 145, 152],
]

a = np.array([[30 + 20 * i + j * 10 + np.random.randint(0, 10) for j in range(200)] for i in range(5)])


def is_feasible(sol, budget):
    return sum(sol) < budget


def value(sol, matrix):
    return sum([matrix[i][sol[i]] for i in range(len(matrix))])


def mkcp_solver(matrix):
    start = time.time()
    matrix = matrix
    res_matrix = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix) + 1)]
    res_matrix[1] = matrix[0]
    sol_matrix = [[{'i_mat': 0, 'i_aux': 0} for i in range(len(matrix[0]))] for j in range(len(matrix))]
    sol_matrix[0] = [{'i_mat': 0, 'i_aux': i} for i in range(len(matrix[0]))]
    # print(time.time() - start)
    # start = time.time()
    for i in range(2, len(matrix) + 1):
        for j in range(len(matrix[0])):
            elems = [[matrix[i - 1][j - k] + res_matrix[i - 1][k], k] for k in range(j + 1)]
            best = max(elems, key=lambda x: x[0])
            # best_mat, best_aux, max_val = best_combination(matrix[i - 1], res_matrix[i - 1])
            sol_matrix[i - 1][j]['i_mat'] = best[1]
            sol_matrix[i - 1][j]['i_aux'] = j-best[1]
            res_matrix[i][j] = best[0]
    sol = []
    # print(time.time() - start)

    k = np.argmax(res_matrix[len(matrix) - 1])
    for i in range(len(matrix) - 1, -1, -1):
        sol.append(sol_matrix[i][k]['i_aux'])
        k = sol_matrix[i][k]['i_mat']
    sol.reverse()

    return sol

#
# start = time.time()
# sol_mkcp = mkcp_solver(a)
# print(f'mkcp({sol_mkcp, value(sol_mkcp, a)}, cost: {sum(i for i in sol_mkcp)}) in {time.time() - start}')
# start = time.time()
# sol_knapsack = knapsack_optimizer(a)
# print(
#     f'knapsack({sol_knapsack, value(sol_knapsack, a)},cost: {sum(i for i in sol_knapsack)}) in {time.time() - start}')
