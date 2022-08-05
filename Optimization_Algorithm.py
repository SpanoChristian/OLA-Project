import numpy as np


def fun(max_value, x):
    return max_value * (1 - np.exp(-x))


def optimization_algorithm(table):
    """
        Algorithm to find the best allocation of the budgets to the various sub-campaigns
        :param table: matrix in which each cell contains the expected return given a specific budget

        Example (given in the slide of the course)
        table = [
            [m_inf, 90, 100, 105, 110, m_inf, m_inf, m_inf],
            [0, 82, 90, 92, m_inf, m_inf, m_inf, m_inf],
            [0, 80, 83, 85, 86, m_inf, m_inf, m_inf],
            [m_inf, 90, 110, 115, 118, 120, m_inf, m_inf],
            [m_inf, 111, 130, 138, 142, 148, 155, m_inf]
        ]

        :return: best allocation, i.e., how to allocate the budget to the various sub-campaigns
    """

    # Budgets
    budgets = np.linspace(0.0, 7.0, 8)
    # print(f"budgets: {budgets}")

    # Get, dynamically, the number of rows and cols
    table_rows, table_row_cols = table.shape

    n_budgets = budgets.size  # How many budgets do we have
    n_campaigns = table_rows  # How many campaigns do we have

    # print(f"Num budgets = {n_budgets}")
    # print(f"Num campaigns = {n_campaigns}")
    # print("--------------------------------")

    # Initializations
    prev_campaign = np.empty(n_budgets)
    tmp_campaign = np.zeros(n_budgets)
    opt_indexes = [[] for i in range(0, n_campaigns - 1)]
    opt_table = [[] for i in range(0, n_campaigns)]

    # print(f"Optimal indexes: {opt_indexes}")

    for i in range(0, n_campaigns + 1):
        curr_campaign = np.zeros(n_budgets)
        if i == 0:
            curr_campaign = tmp_campaign
        elif i == 1:
            curr_campaign = table[i - 1, :]
            opt_table[i - 1] = np.append(opt_table[i - 1], curr_campaign)
        else:
            tmp_campaign = table[i - 1, :]
            for j in range(0, n_budgets):
                opt_index = 0
                tmp_max_sum_n_el = j + 1
                tmp_max_sum = np.empty(j + 1)
                pos_budget_prev_campaign = [x for x in range(len(budgets)) if budgets[x] <= budgets[j]]
                if j == 0:
                    tmp_max_sum[j] = tmp_campaign[np.max(pos_budget_prev_campaign)] + prev_campaign[0]
                else:
                    for k in range(0, j + 1):
                        pos_budget_tmp_campaign = np.max(pos_budget_prev_campaign) - k
                        tmp_max_sum[k] = tmp_campaign[pos_budget_tmp_campaign] + prev_campaign[k]
                curr_campaign[j] = np.max(tmp_max_sum)
                opt_index = np.where(tmp_max_sum == curr_campaign[j])
                opt_indexes[i - 2] = np.append(opt_indexes[i - 2], tmp_max_sum_n_el - np.max(opt_index).astype(int) - 1)

            opt_table[i - 1] = np.append(opt_table[i - 1], curr_campaign)
        prev_campaign = curr_campaign

        # print(f"Campaign c_{i}: {curr_campaign}")

    # print(f"Optimal table: {opt_table}")

    # Subtracting the corresponding budget to the optimal
    for k in range(0, n_budgets):
        curr_campaign[k] -= budgets[k]

    # print(f"\n\nSubtracting Budget: {curr_campaign}")
    idx_best_budget = int(max((v, i) for i, v in enumerate(curr_campaign))[1])

    # print(f"B* (Best Budget) = {int(budgets[idx_best_budget])}")
    # print(f"Algorithm complexity = {n_campaigns * n_budgets ^ 2}")

    allocations = [0 for r in range(n_campaigns)]
    for i in range(n_campaigns - 1, 0, -1):
        subc_col = int(opt_indexes[i - 1][idx_best_budget])
        allocations[i] = subc_col
        idx_best_budget -= subc_col

    allocations[0] = idx_best_budget

    tot_clicks = np.array([])
    for i in range(0, len(allocations)):
        tot_clicks = np.append(tot_clicks, table[i][allocations[i]])
        # print(f"Budget for c_{i + 1}: {allocations[i]}K € -- Number of clicks: {tot_clicks[i]}")

    return allocations

    # print("---------------------------------------------------------")
    # print(f"\t\t  Sum = {np.sum(allocations)}K €\t\t\t    Sum = {np.sum(tot_clicks)}")
    # print(f"\nBest allocation: {allocations}")