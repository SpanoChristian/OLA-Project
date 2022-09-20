from Environments.Base_Environment import *
from Environments.Environment_step5 import *
from Learners.GPTS_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils.MKCP import *
from utils.utils import *
from Runner import *

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
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 2, 20)

env = Environment5(n_subcampaigns=config.n_subcampaigns,
                   subcampaign_class=Subcampaign5,
                   alpha_bars=config.alpha_bars,
                   multiplier=10000,
                   speeds=config.speeds,
                   opponent=config.opponent,
                   adj_matrix=config.adj_matrix,
                   sigma_matrix=0.00000001,
                   budgets=config.budgets,
                   daily_clicks=100
                   )

runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=GPTS_Learner)

start = time.time()
T = 40
runner.run(T)

gs = gridspec.GridSpec(1, 2)
plt.figure(figsize=(13, 5))

plt.subplot(gs[0, 0])
best_arms = mkcp_solver(np.array(env.round()))
rewards = env.compute_rewards(best_arms)
clairvoyant_mean, clairvoyant_cb, sol = get_clairvoyant_score(env, 5)
y_clairvoyant = np.full(T, clairvoyant_mean)
y_clairvoyant_cb = np.full((T, 2), clairvoyant_cb)

x = range(T)
y = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in range(T)]

plt.plot(x, y, label=u'GPTS reward')
plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')
plt.fill_between(x, y_clairvoyant_cb[:, 0], y_clairvoyant_cb[:, 1], alpha=.2, fc='r', ec='None',
                 label='95% conf interval')

plt.text(x=T * 0.75, y=np.max(y_clairvoyant) * 0.9, s=f'total time: {"{:.2f}".format(time.time() - start)}')

plt.subplot(gs[0, 1])
regret = []
for i in range(len(x)):
    diff = y_clairvoyant_cb[i][0] - y[i]
    aux = (diff if diff > 0 else 0) + (regret[-1] if i > 0 else 0)
    regret.append(aux)

plt.plot(x, regret)
plt.text(x=T * 0.6, y=regret[-1] * 0.5, s=f'total regret: {"{:.2f}".format(regret[-1])}\n'
                                          f'total clicks: {"{:.2f}".format(sum(y))}')

plt.show()
count = [runner.pulled_super_arms.count(i) for i in runner.pulled_super_arms]
lower_bounds = []
for i in range(T):
    lower_bounds.append([])
    for j in range(env.n_subcampaigns):
        pulled = np.array(runner.pulled_super_arms)[:, j]
        mean = np.mean([y[k] for k in range(len(pulled)) if pulled[k] == pulled[i]])
        div = 2 * np.count_nonzero(pulled == pulled[i])
        lower_bounds[i].append(mean - np.sqrt(-np.log(0.999) / div))

res = [sum(i) for i in lower_bounds]
print(np.array(res))
print(np.array(list(zip(res, runner.pulled_super_arms))))
_max = max(list(zip(res, runner.pulled_super_arms)), key=lambda x: x[0])
print(f'best arm computed: {_max[1]}, cumulative_lb: {_max[0]}, reward: {y[res.index(_max[0])]}')
print(f'highest score reached: {max(y)}, highest score arm: {runner.pulled_super_arms[y.index(max(y))]}')
print(f'optimal arm: {env.optimal_sol}, optimal arm reward: {env.optimal_sol_reward}')

best_arm_computed = _max[1]
highest_score_arm = runner.pulled_super_arms[y.index(max(y))]
clairvoyant = env.optimal_sol
records = [[], [], []]

for i in range(100):
    records[0].append(sum(env.compute_rewards(best_arm_computed)))
    records[1].append(sum(env.compute_rewards(highest_score_arm)))
    records[2].append(sum(env.compute_rewards(clairvoyant)))

scores = [np.mean(i) for i in records]
print(scores)
