import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
import logging
import warnings
from Environments.Environment_step6_SW import *
from Learners import SW_Learner
from Learners.GPUCB_Learner import *
from Learners.SW_Learner import *
from Runner import Runner
from utils.MKCP import mkcp_solver

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
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 1.7, 5)

window_factor = 2
T = 1000
window_size = window_factor * int(np.sqrt(T))
# rows: phases, cols: arms
p = np.array([np.ones(100),
              np.random.rand(100),
              np.ones(100)])

env = Base_Environment(n_subcampaigns=config.n_subcampaigns,
                       subcampaign_class=Subcampaign6,
                       alpha_bars=config.alpha_bars,
                       speeds=config.speeds,
                       opponent=config.opponent,
                       adj_matrix=config.adj_matrix,
                       budgets=config.budgets,
                       daily_clicks=100,
                       # probs_matrix=p,
                       # horizon=T
                       )

runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=SW_Learner, dont_update_before=1)

start = time.time()
runner.run(T)

gs = gridspec.GridSpec(1, 2)
plt.figure(figsize=(13, 5))

plt.subplot(gs[0, 0])
best_arms = mkcp_solver(np.array(env.round()))
rewards = env.compute_rewards(best_arms)
y_clairvoyant = [sum([rewards[j] for j in range(env.n_subcampaigns)]) for i in range(T)]

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in range(T)]

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
