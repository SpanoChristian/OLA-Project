from Environments.Base_Environment import *
from Environments.Environment_step6 import *
from Learners.GPTS_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils.MKCP import *
from Runner import *
from ComparisonRunner import *
from Learners.GPUCB_Learner import *
from Learners.CD_GPUCB import *

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
config.speeds = [np.random.uniform(0.001, 40) for i in range(config.n_subcampaigns)]
config.opponent = config.ratios[0]
config.adj_matrix = np.array([
    [0, 0, 0.04, 0.07, 0.9],
    [0, 0, 0.03, 0.03, 0.7],
    [0, 0, 0, 0.2, 0.5],
    [0.02, 0.02, 0, 0, 0],
    [0, 0.01, 0, 0, 0]
])
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 1.2, 10)

env = Environment6(n_subcampaigns=config.n_subcampaigns,
                   subcampaign_class=Subcampaign5,
                   alpha_bars=config.alpha_bars,
                   multiplier=100000,
                   speeds=config.speeds,
                   opponent=config.opponent,
                   adj_matrix=config.adj_matrix,
                   sigma_matrix=0.001,
                   budgets=config.budgets,
                   daily_clicks=100,
                   phase=40
                   )

# runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=GPTS_Learner)
# [CD_GPUCB_Learner(env.budgets) for _ in range(env.n_subcampaigns)],
learners = [
            [GPTS_Learner(env.budgets) for _ in range(env.n_subcampaigns)]]

runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=GPTS_Learner)
start = time.time()
T = 100
runner.run(T)

gs = gridspec.GridSpec(1, 2)
plt.figure(figsize=(13, 5))

plt.subplot(gs[0, 0])
y_clairvoyant = [phase['reward']
                 for i, phase in enumerate(env.optimal)
                 for _ in
                 range((env.optimal[i + 1]['start_from'] if i < len(env.optimal) - 1 else T) - phase['start_from'])
                 ]
x = range(T)
y = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in range(T)]
# y2 = [sum([learner.collected_rewards[i] for learner in runner.learners[1]]) for i in range(T)]

plt.plot(x, y, color='blue', label=u'GPTS reward')
# plt.plot(x, y2, color='red', label=u'GPTS reward')

# changes = []
# for learner in runner.learners[0]:
#     for change in learner.changes:
#         changes.append(change)
# if len(changes) > 0:
#     count = np.array([[change, float(changes.count(change))] for change in list(set(changes))])
#     count[:, 1] = count[:, 1]/float(np.linalg.norm(count[:, 1]))
#     for item in count:
#         plt.axvline(item[0], color=[0, 150/255.0, 0, item[1]])
#
plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')

plt.text(x=T * 0.75, y=np.max(y_clairvoyant) * 0.9, s=f'total time: {"{:.2f}".format(time.time() - start)}')

plt.subplot(gs[0, 1])
regret = []
regret2 = []
for i in range(len(x)):
    aux = y_clairvoyant[i] - y[i] + (regret[-1] if i > 0 else 0)
    regret.append(aux)
    # aux = y_clairvoyant[i] - y2[i] + (regret2[-1] if i > 0 else 0)
    # regret2.append(aux)

plt.plot(x, regret, color='blue')
# plt.plot(x, regret2, color='red')

plt.text(x=T * 0.6, y=regret[-1] * 0.5, s=f'total regret: {"{:.2f}".format(regret[-1])}\n'
                                          f'total clicks: {"{:.2f}".format(sum(y))}')

plt.show()
