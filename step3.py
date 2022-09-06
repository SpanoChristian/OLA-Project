from Environments.Base_Environment import *
from Learners.GPTS_Learner import *
import logging
from utils.knapsack import *
import warnings
import matplotlib.gridspec as gridspec
from utils.Optimization_Algorithm import *
import matplotlib.pyplot as plt
from utils.config import *
import time
from utils.MKCP import *

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


class Config(object):
    pass


config = Config()
config.n_subcampaigns = 5
config.ratios = to_sum_1(np.array([3, 2, 3, 4, 5, 6]))
config.alpha_bars = config.ratios[1:]
config.speeds = [np.random.uniform(0.1, 0.9) for i in range(config.n_subcampaigns)]
config.opponent = config.ratios[0]
config.adj_matrix = np.array([
    [0, 0, 0.04, 0.07, 0.9],
    [0, 0, 0.03, 0.03, 0.7],
    [0, 0, 0, 0.2, 0.5],
    [0.02, 0.02, 0, 0, 0],
    [0, 0.01, 0, 0, 0]
])
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 2, 300)

env = Base_Environment(n_subcampaigns=config.n_subcampaigns,
                       alpha_bars=config.alpha_bars,
                       speeds=config.speeds,
                       opponent=config.opponent,
                       adj_matrix=config.adj_matrix,
                       budgets=config.budgets,
                       daily_clicks=100
                       )

config.dont_update_before = 1

learners = []
for i in range(0, len(config.alpha_bars)):
    learners.append(GPTS_Learner(arms=config.budgets))

T = 40
x = [[] for i in range(env.n_subcampaigns)]
y = [[] for i in range(env.n_subcampaigns)]
y_pred = [[] for i in range(env.n_subcampaigns)]
x_pred = config.budgets
sigmas = [[] for i in range(env.n_subcampaigns)]
gs = gridspec.GridSpec(1, env.n_subcampaigns)
start = time.time()
for i in range(0, T):
    samples = np.zeros(shape=(0, len(config.budgets)))
    for learner in learners:
        tmp = np.array(learner.pull_all_arms())
        samples = np.append(samples, [tmp], axis=0)

    arms = mkcp_solver(samples)
    env.compute_rewards(arms)

    for j in range(config.n_subcampaigns):
        arm_reward = env.get_reward(subcampaign=j)

        x[j].append(arms[j])
        y[j].append(arm_reward)

        if i < config.dont_update_before:
            learners[j].update_observations(arms[j], arm_reward)
        else:
            learners[j].update(arms[j], arm_reward)

gs = gridspec.GridSpec(1, 2)
plt.figure(figsize=(13, 5))

plt.subplot(gs[0, 0])
best_arms = mkcp_solver(np.array(env.round()))
env.compute_rewards(best_arms)
y_clairvoyant = [sum([env.get_reward(j) for j in range(config.n_subcampaigns)]) for i in range(T)]

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in learners]) for i in range(T)]

plt.plot(x, y, label=u'GPTS reward')
plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')

plt.text(x=T * 0.75, y=np.max(y_clairvoyant) * 0.9, s=f'total time: {"{:.2f}".format(time.time() - start)}')

plt.subplot(gs[0, 1])
regret = []
for i in range(len(x)):
    aux = y_clairvoyant[i] - y[i] + (regret[-1] if i > 0 else 0)
    regret.append(aux)

plt.plot(x, regret)
plt.text(x=T * 0.6, y=regret[-1] * 0.5, s=f'total regret: {"{:.2f}".format(regret[-1])}\n'
                                                         f'total clicks: {"{:.2f}".format(sum(y))}')

plt.show()
