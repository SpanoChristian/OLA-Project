import numpy as np
from Environment_step3 import *
from GPTS_Learner import *
import logging

logging.basicConfig(level=logging.DEBUG)


def fun(x, x_bar, speed):
    return x_bar * (1.0 - np.exp(-x * speed))


logging.debug("setting up parameters")
sub_campaigns = 5
alpha_bar = np.array(range(1, sub_campaigns + 1))
speeds = np.random.normal([1, 1, 1, 1, 1], 0.2)
sigmas = np.array([3 for i in range(sub_campaigns)])
max_budget = max(speeds)  # half the maximum required to reach the saturation of all the arms
n_arms = 6
arms = np.linspace(0.0, speeds, n_arms)

logging.debug(f'''
number of subcampaigns:     {speeds},
alpha_bars:                 {alpha_bar},
speeds:                     {speeds},
sigmas:                     {sigmas},
max_budget:                 {max_budget},
n_arms:                     {n_arms},
arms:                       {arms}
''')

env = Environment()

for i in range(sub_campaigns):
    sc = Subcampaign(budgets=arms[i], function=lambda x: fun(x, alpha_bar[i], speeds[i]), sigma=sigmas[i])
    env.add_subcampaign(subcampaign=sc)

print("created subcampaigns")

learners = []
for i in range(0, len(alpha_bar)):
    learners.append(GPTS_Learner(arms=arms, n_arms=n_arms))


def greedy_knapsack(table, budget):
    end = False
    res = np.zeros(table.shape[0])
    while not end:
        i_max = np.unravel_index(np.argmax(table), table.shape)
        if table[i_max] < budget:
            res[i_max[0]] = i_max[1]
            budget -= table[i_max]
            table = np.delete(table, i_max[0], 0)
        else:
            table[i_max] = 0
        if table.shape[0] == 0 or np.min(table) > budget:
            end = True

    return res


m = np.array([[1, 2, 3], [4, 6, 7], [7, 8, 9]])
greedy_knapsack(m, 14)

T = 60
for i in range(0, T):
    samples = np.array([])
    for learner in learners:
        np.append(samples, (np.array(learner.pull_all_arms())))

    arms = greedy_knapsack(m, max_budget)

    for j in range(sub_campaigns):
        arm_reward = env.round(subcampaign=j, pulled_arm=arms[i])
        learners[j].update_observations(j, arm_reward)
