import time

from Environments.Environment_ac import *
from Learners.GPTS_Learner import *
from Learners.GPUCB_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
from utils.Optimization_Algorithm import *
import matplotlib.pyplot as plt
from utils.kanpsack import *

plt.ion()

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")


# alpha function to be learnt
def fun(x, x_bar, speed):
    return x_bar * (1.0 - np.exp(-x * speed))


def _time(function):
    _start = time.time()
    res = function()
    return res, time.time() - _start


logging.debug("setting up parameters")


class Config(object):
    pass


config = Config()
config.sub_campaigns = 5
config.alpha_bar = np.array([i * 100 + 500 for i in range(1, config.sub_campaigns + 1)])
config.speeds = np.random.uniform(0.1, 0.9, 5)
config.sigmas = np.array([0 for i in config.alpha_bar])
config.max_budget = 4
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

env = Environment(adj_matrix=config.adj_matrix, matrix_sigma=0.02, daily_clicks=1000,
                  alpha_bar_high=15000000, alpha_bar_low=100000, speed_high=0.9, speed_low=0.1,
                  opponent_mean=1200000, opponent_variance=30)

for i in range(config.sub_campaigns):
    sc = Subcampaign(budgets=config.arms, alpha_bar=config.alpha_bar[i], speed=config.speeds[i])
    env.add_subcampaign(subcampaign=sc)

print("created subcampaigns")

learners = []
for i in range(0, len(config.alpha_bar)):
    learners.append(GPTS_Learner(arms=config.arms, n_arms=config.n_arms))


def compute_clairvoyant_reward():
    clairvoyant_solution = knapsack_optimizer(np.array(env.round()))
    k = [np.random.normal(env.opponent_mean, env.opponent_variance)]
    env.next_day(clairvoyant_solution)
    k.extend([env.get_reward(subcampaign) for subcampaign in range(config.sub_campaigns)])
    clairvoyant_reward = sum(k[1:])
    return clairvoyant_reward


T = 151
x = [[] for i in range(5)]
y = [[] for i in range(5)]
y_clairvoyant = []
y_pred = [[] for i in range(5)]
x_pred = config.arms
sigmas = [[] for i in range(5)]
differences = [0 for i in range(config.sub_campaigns)]

start = time.time()
aux = time.time()
time_learning = 0
time_optimizer = 0
gs = gridspec.GridSpec(1, 5)

for i in range(0, T):
    if i % 50 == 0:
        print(f'iteration: {i}, time since last: {time.time() - aux}, total time: {time.time() - start}')
        aux = time.time()

    samples = np.zeros(shape=(0, config.n_arms))
    for learner in learners:
        tmp = np.array(learner.pull_all_arms())
        samples = np.append(samples, [tmp], axis=0)

    arms, _time_optimizer = _time(lambda: knapsack_optimizer(samples))
    time_optimizer += _time_optimizer

    if i % env.daily_clicks == 0:
        y_clairvoyant.append(compute_clairvoyant_reward())
    else:
        y_clairvoyant.append(y_clairvoyant[len(y_clairvoyant) - 1])

    env.next_day(arms)
    for j in range(config.sub_campaigns):
        arm_reward = env.get_reward(j)

        x[j].append(config.arms[arms[j]])
        y[j].append(arm_reward)

        time_learning += _time(lambda: learners[j].update(arms[j], arm_reward))[1]

    if i % 50 == 0:
        print(f'time_opt: {time_optimizer}, time_learning: {time_learning}')

    if i % 10 == 0:
        plt.figure(figsize=(20, 5))
        for j in range(config.sub_campaigns):
            y_pred[j] = learners[j].means
            sigmas[j] = learners[j].sigmas

            plt.subplot(gs[0, j])


            def n(k):
                e = [1200000]
                e.extend([i.alpha_bar for i in env.subcampaigns])
                norm_alpha_bar = np.linalg.norm(e)

                return fun(k, x_bar=env.subcampaigns[j].alpha_bar/norm_alpha_bar*env.daily_clicks,
                           speed=env.subcampaigns[j].speed)


            plt.scatter(x[j], y[j], s=5, label=u'Observed Clicks')
            plt.plot(x_pred, n(np.array(x_pred)), 'r', label=u'Alpha')
            plt.plot(x_pred, y_pred[j], 'b-', label=u'Predicted Clicks')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([y_pred[j] - 1.96 * sigmas[j], (y_pred[j] + 1.96 * sigmas[j])[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc="lower right")
        plt.show()

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in learners]) for i in range(T)]

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
plt.plot(x, y, label=u'GPTS reward')
plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')

# plt.legend(loc="upper left")
plt.show()
