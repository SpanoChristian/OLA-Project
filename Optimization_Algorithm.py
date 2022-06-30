import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
feature_labels = ["1", "2", "3"]
budgets = np.linspace(0, 10, num=11)
x = np.linspace(0, max(budgets), num=550)
colors = ['r', 'g', 'b']

plt.figure(figsize=(14,8))

for i in enumerate(feature_labels):
    y = []
    y.append(100*(1-np.exp(x)))
    scatter = []
    scatter.append(100*(1-np.exp(budgets)))
    aggr_y = sum(y)
    aggr_scatter = sum(scatter)

    plt.plot(x, aggr_y, color=colors[i], label=feature_labels[i])
    plt.scatter(budgets, aggr_scatter, color=colors[i])
    plt.title("Click Functions aggregated")
    plt.xlabel("Budget (€)")
    plt.ylabel("Number of Clicks")
    plt.legend()
'''


def fun(max_value, x):
    return max_value * (1 - np.exp(-x))


def min_max_budget(min_budget, max_budget, fun_budgets):
    for i in range(0, min_budget):
        fun_budgets[i] = -np.inf

    for j in range(max_budget, len(fun_budgets)):
        fun_budgets[j] = -np.inf

    return fun_budgets


budgets = np.linspace(0, 10, num=11)
x = np.linspace(0, max(budgets), num=550)
print(budgets)

# Vector of budgets => we will have to sort it
# budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])
budgets = np.linspace(0.0, 7.0, 8)

# Generate the table setting the min and max budgets
# A budget of a given campaign has a lower and upper bound that are defined by the function 'min_max_budget'
# For all the budgets lower or higher than the one specified as parameter, the value is -inf
fun_11 = min_max_budget(0, 10, fun(200, budgets))
fun_12 = min_max_budget(0, 10, fun(180, budgets))
fun_13 = min_max_budget(0, 10, fun(300, budgets))
fun_14 = min_max_budget(0, 10, fun(120, budgets))
fun_15 = min_max_budget(0, 10, fun(140, budgets))

fun_21 = min_max_budget(0, 10, fun(300, budgets))
fun_22 = min_max_budget(0, 10, fun(200, budgets))
fun_23 = min_max_budget(0, 10, fun(250, budgets))
fun_24 = min_max_budget(0, 10, fun(180, budgets))
fun_25 = min_max_budget(0, 10, fun(230, budgets))

fun_31 = min_max_budget(0, 10, fun(150, budgets))
fun_32 = min_max_budget(0, 10, fun(300, budgets))
fun_33 = min_max_budget(0, 10, fun(120, budgets))
fun_34 = min_max_budget(0, 10, fun(110, budgets))
fun_35 = min_max_budget(0, 10, fun(100, budgets))

# Here we assume to have it (=STEP 1)
value_budget = np.array([fun_11,
                         fun_12,
                         fun_13,
                         fun_14,
                         fun_15])

#m_inf = -np.inf
'''value_budget = np.array([[m_inf, 90, 100, 105, 110, m_inf, m_inf, m_inf],
            [0, 82, 90, 92, m_inf, m_inf, m_inf, m_inf],
            [0, 80, 83, 85, 86, m_inf, m_inf, m_inf],
            [m_inf, 90, 110, 115, 118, 120, m_inf, m_inf],
            [m_inf, 111, 130, 138, 142, 148, 155, m_inf]])'''

# Get, dynamically, the number of rows and cols
value_budget_rows, value_budget_cols = value_budget.shape

n_budgets = budgets.size            # How many budgets do we have
n_campaigns = value_budget_rows     # How many campaigns do we have

print(f"Num budgets = {n_budgets}")
print(f"Num campaigns = {n_campaigns}")
print("--------------------------------")

# Initializations
prev_campaign = np.empty(n_budgets)
tmp_campaign = np.zeros(n_budgets)
opt_indexes = [[] for i in range(0, n_campaigns-1)]
opt_table = [[] for i in range(0, n_campaigns)]

print(f"Optimal indexes: {opt_indexes}")

for i in range(0, n_campaigns + 1):
    curr_campaign = np.zeros(n_budgets)
    if i == 0:
        curr_campaign = tmp_campaign
    elif i == 1:
        curr_campaign = value_budget[i-1, :]
        opt_table[i - 1] = np.append(opt_table[i - 1], curr_campaign)
    else:
        tmp_campaign = value_budget[i-1, :]
        for j in range(0, n_budgets):
            opt_index = 0
            tmp_max_sum_n_el = j + 1
            tmp_max_sum = np.empty(j+1)
            pos_budget_prev_campaign = [x for x in range(len(budgets)) if budgets[x] <= budgets[j]]
            if j == 0:
                tmp_max_sum[j] = tmp_campaign[np.max(pos_budget_prev_campaign)] + prev_campaign[0]
            else:
                for k in range(0, j+1):
                    pos_budget_tmp_campaign = np.max(pos_budget_prev_campaign) - k
                    tmp_max_sum[k] = tmp_campaign[pos_budget_tmp_campaign] + prev_campaign[k]
            curr_campaign[j] = np.max(tmp_max_sum)
            opt_index = np.where(tmp_max_sum == curr_campaign[j])
            opt_indexes[i-2] = np.append(opt_indexes[i-2], tmp_max_sum_n_el - np.max(opt_index).astype(int) - 1)

        opt_table[i-1] = np.append(opt_table[i-1], curr_campaign)
    prev_campaign = curr_campaign

    print(f"Campaign c_{i}: {curr_campaign}")

print(f"Optimal table: {opt_table}")

# Subtracting the corresponding budget to the optimal
for k in range(0, n_budgets):
    curr_campaign[k] -= budgets[k]

print(f"\n\nSubtracting Budget: {curr_campaign}")
idx_best_budget = int(max((v, i) for i, v in enumerate(curr_campaign))[1])

print(f"B* (Best Budget) = {int(budgets[idx_best_budget])}")
print(f"Algorithm complexity = {n_campaigns*n_budgets^2}")

allocations = [0 for r in range(n_campaigns)]
for i in range(n_campaigns - 1, 0, -1):
    subc_col = int(opt_indexes[i-1][idx_best_budget])
    allocations[i] = subc_col
    idx_best_budget -= subc_col

allocations[0] = idx_best_budget

tot_clicks = np.array([])
for i in range(0, len(allocations)):
    tot_clicks = np.append(tot_clicks, opt_table[i][allocations[i]])
    print(f"Budget for c_{i+1}: {allocations[i]}K € -- Number of clicks: {tot_clicks[i]}")

print("---------------------------------------------------------")
print(f"\t\t  Sum = {np.sum(allocations)}K €\t\t\t    Sum = {np.sum(tot_clicks)}")
print(f"\nBest allocation: {allocations}")

gs = gridspec.GridSpec(1, 3)
plt.figure()
plt.subplot(gs[0, 0])
plt.plot(budgets, fun_11, 'red')
plt.plot(budgets, fun_12, 'green')
plt.plot(budgets, fun_13, 'blue')
plt.plot(budgets, fun_14, 'cyan')
plt.plot(budgets, fun_15, 'orange')
plt.legend(["Product 1", "Product 2", "Product 3", "Product 4", "Product 5"], loc="lower right")
plt.xlabel("Budget")
plt.ylabel("Value * #clicks")
plt.title("User Class 1")

plt.subplot(gs[0, 1])
plt.plot(budgets, fun_21, 'red')
plt.plot(budgets, fun_22, 'green')
plt.plot(budgets, fun_23, 'blue')
plt.plot(budgets, fun_24, 'cyan')
plt.plot(budgets, fun_25, 'orange')
plt.legend(["Product 1", "Product 2", "Product 3", "Product 4", "Product 5"], loc="lower right")
plt.xlabel("Budget")
plt.ylabel("Value * #clicks")
plt.title("User Class 2")

plt.subplot(gs[0, 2])
plt.plot(budgets, fun_31, 'red')
plt.plot(budgets, fun_32, 'green')
plt.plot(budgets, fun_33, 'blue')
plt.plot(budgets, fun_34, 'cyan')
plt.plot(budgets, fun_35, 'orange')
plt.legend(["Product 1", "Product 2", "Product 3", "Product 4", "Product 5"], loc="lower right")
plt.xlabel("Budget")
plt.ylabel("Value * #clicks")
plt.title("User Class 3")

plt.show()