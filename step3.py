from Environments.Environment_step3 import *
from Learners.GPTS_Learner import *
import logging
from utils.kanpsack import *
import warnings
import matplotlib.gridspec as gridspec
from utils.Optimization_Algorithm import *
import matplotlib.pyplot as plt

plt.ion()

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")


# alpha function to be learnt
def fun(x, x_bar, speed):
    return x_bar * (1.0 - np.exp(-x * speed))


logging.debug("setting up parameters")


class Config(object):
    pass


config = Config()
config.sub_campaigns = 5
config.alpha_bar = np.array([i + 10 for i in range(1, config.sub_campaigns + 1)])
config.speeds = np.random.normal([0.5, 0.3, 0.4, 0.5, 0.6], 0.0)
config.sigmas = np.array([0.05 * (i + 1) for i in range(config.sub_campaigns)])
config.max_budget = sum(config.speeds) * 5
config.n_arms = 10
config.arms = np.linspace(0.0, config.max_budget, config.n_arms).T

config.adj_matrix = np.array([[0, 0, 0.2, 0, 0],
                              [0.1, 0, 0, 0.3, 0],
                              [0, 0.2, 0, 0.1, 0],
                              [0.2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]
                              ])
# p2 -> p1, p4
# p1 -> p3
#
config.lambda_param = 0.5
config.second_secondary = np.array([[0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])



m = np.array(0)
for i in range(config.sub_campaigns):
    for j in range(config.n_arms):
        print(fun(config.arms[j], config.alpha_bar[i], config.speeds[i]), ',', end='')
    print('\n')

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
    sc = Subcampaign(budgets=config.arms, function=lambda x: fun(x, config.alpha_bar[i], config.speeds[i]),
                     sigma=config.sigmas[i])
    env.add_subcampaign(subcampaign=sc)

print("created subcampaigns")

learners = []
for i in range(0, len(config.alpha_bar)):
    learners.append(GPTS_Learner(arms=config.arms, n_arms=config.n_arms))


aux = []
for i in range(0, 100):
    k = np.array(env.round())
    sol = knapsack_optimizer(np.array(env.round()))
    aux.append(sol)

list = [{'elem': item, 'count': aux.count(item)} for item in aux]
max = max(range(len(list)), key=lambda i: list[i]['count'])
clairvoyant_solution = list[max]['elem']

clairvoyant_reward = sum(np.random.dirichlet([env.round(subcampaign, arm) for subcampaign, arm in zip(range(config.sub_campaigns),
                                                                                  clairvoyant_solution)]))

T = 101
x = [[] for i in range(5)]
y = [[] for i in range(5)]
y_pred = [[] for i in range(5)]
x_pred = config.arms
sigmas = [[] for i in range(5)]
gs = gridspec.GridSpec(1, 5)

for i in range(0, T):
    samples = np.zeros(shape=(0, config.n_arms))
    for learner in learners:
        tmp = np.array(learner.pull_all_arms())
        samples = np.append(samples, [tmp], axis=0)

    arms = optimization_algorithm(samples)

    for j in range(config.sub_campaigns):

        arm_reward = env.round(subcampaign=j, pulled_arm=arms[j])

        x[j].append(arms[j])
        y[j].append(arm_reward)

        if i < 50 and i % 10 != 0:
            learners[j].update_observations(arms[j], arm_reward)
        else:
            learners[j].update(arms[j], arm_reward)

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in learners]) for i in range(T)]

plt.plot(x, y, label=u'GPTS reward')
plt.plot(x, [clairvoyant_reward for i in range(T)], label=u'Clairvoyant reward')
plt.show()

# if i % 10 == 0:
#     plt.figure(figsize=(20, 5))
#     for j in range(config.sub_campaigns):
#         y_pred[j] = learners[j].means
#         sigmas[j] = learners[j].sigmas
#
#         plt.subplot(gs[0, j])
#
#
#         def n(k):
#             return fun(k, x_bar=config.alpha_bar[j], speed=config.speeds[j])
#
#
#         plt.scatter(x[j], y[j], s=5, label=u'Observed Clicks')
#         plt.plot(x_pred, n(np.array(x_pred)), 'r', label=u'Alpha')
#         plt.plot(x_pred, y_pred[j], 'b-', label=u'Predicted Clicks')
#         plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
#                  np.concatenate([y_pred[j] - 1.96 * sigmas[j], (y_pred[j] + 1.96 * sigmas[j])[::-1]]),
#                  alpha=.5, fc='b', ec='None', label='95% conf interval')
#         plt.xlabel('$x$')
#         plt.ylabel('$n(x)$')
#         plt.legend(loc="lower right")
#     plt.draw()
#     plt.pause(0.02)

plt.ioff()
plt.show()

print('ciao')
