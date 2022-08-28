from Environments.Base_Environment import *
from Learners.GPTS_Learner import *
import logging
from utils.kanpsack import *
import warnings
import matplotlib.gridspec as gridspec
from utils.Optimization_Algorithm import *
import matplotlib.pyplot as plt
from utils.config import *

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
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 1.5, 200)

env = Base_Environment(n_subcampaigns=config.n_subcampaigns,
                       alpha_bars=config.alpha_bars,
                       speeds=config.speeds,
                       opponent=config.opponent,
                       adj_matrix=config.adj_matrix,
                       budgets=config.budgets,
                       daily_clicks=40000
                       )

config.dont_update_before = 3

learners = []
for i in range(0, len(config.alpha_bars)):
    learners.append(GPTS_Learner(arms=config.budgets))

T = 35
x = [[] for i in range(5)]
y = [[] for i in range(5)]
y_pred = [[] for i in range(5)]
x_pred = config.budgets
sigmas = [[] for i in range(5)]
gs = gridspec.GridSpec(1, 5)

for i in range(0, T):
    samples = np.zeros(shape=(0, len(config.budgets)))
    for learner in learners:
        tmp = np.array(learner.pull_all_arms())
        samples = np.append(samples, [tmp], axis=0)

    arms = knapsack_optimizer(samples)
    env.compute_rewards(arms)

    for j in range(config.n_subcampaigns):
        arm_reward = env.get_reward(subcampaign=j)

        x[j].append(arms[j])
        y[j].append(arm_reward)

        if i < config.dont_update_before:
            learners[j].update_observations(arms[j], arm_reward)
        else:
            learners[j].update(arms[j], arm_reward)

best_arms = knapsack_optimizer(np.array(env.round()))
env.compute_rewards(best_arms)
y_clairvoyant = [sum([env.get_reward(j) for j in range(config.n_subcampaigns)]) for i in range(T)]

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in learners]) for i in range(T)]

plt.plot(x, y, label=u'GPTS reward')
plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')
plt.show()