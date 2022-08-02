import numpy as np
from Environment_step3 import *
from GPTS_Learner import *
import logging

logging.basicConfig(level=logging.DEBUG)


# alpha function to be learnt
def fun(x, x_bar, speed):
    return x_bar * (1.0 - np.exp(-x * speed))


logging.debug("setting up parameters")


class Config(object):
    pass


config = Config()
config.sub_campaigns = 5
config.alpha_bar = np.array(range(1, config.sub_campaigns + 1))
config.speeds = np.random.normal([1, 1, 1, 1, 1], 0.2)
config.sigmas = np.array([3 for i in range(config.sub_campaigns)])
config.max_budget = sum(config.speeds) * 5 / 2  # half the maximum required to reach the saturation of all the arms
config.n_arms = 6
config.arms = np.linspace(0.0, config.speeds, config.n_arms).T

logging.debug(f'''
number of subcampaigns:     {config.speeds},
alpha_bars:                 {config.alpha_bar},
speeds:                     {config.speeds},
sigmas:                     {config.sigmas},
max_budget:                 {config.max_budget},
n_arms:                     {config.n_arms},
arms:                       {config.arms}
''')

env = Environment()

for i in range(config.sub_campaigns):
    sc = Subcampaign(budgets=config.arms[i], function=lambda x: fun(x, config.alpha_bar[i], config.speeds[i]),
                     sigma=config.sigmas[i])
    env.add_subcampaign(subcampaign=sc)

print("created subcampaigns")

learners = []
for i in range(0, len(config.alpha_bar)):
    learners.append(GPTS_Learner(arms=config.arms, n_arms=config.n_arms))


def greedy_knapsack(original_table, budget):
    if sum([np.min(x) for x in original_table]) > budget:
        raise Exception("Impossible optimization problem")

    table = original_table.copy()
    res = [np.argmin(x) for x in table]
    budget -= sum([x[res[i]] for i, x in enumerate(table)])
    while budget >= 0 and np.max(table) >= 0:
        best = np.unravel_index(np.argmin(abs(table - budget)), table.shape)
        complementary = table[best] - table[best[0]][res[best[0]]]
        if complementary <= budget and best[1] != res[best[0]]:
            budget -= complementary
            res[best[0]] = best[1]

        table[best[0]] = np.ones(table.shape[1]) * float('-inf')

    return res, sum([x[res[i]] for i, x in enumerate(original_table)])


# m = np.array([[1, 2, 3], [4, 6, 7], [7, 8, 9]])
# res, total = greedy_knapsack(m, 14)

T = 60
for i in range(0, T):
    samples = np.zeros(shape=(0, config.n_arms))
    for learner in learners:
        tmp = np.array(learner.pull_all_arms())
        samples = np.append(samples, [tmp], axis=0)

    arms, total = greedy_knapsack(samples.copy(), config.max_budget)

    for j in range(config.sub_campaigns):
        arm_reward = env.round(subcampaign=j, pulled_arm=config.arms[i])
        learners[j].update_observations(j, arm_reward)
