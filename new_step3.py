import numpy as np
from Environment_step3 import *
from GPTS_Learner import *
import logging
from kanpsack import *
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure

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
config.alpha_bar = np.array(range(1, config.sub_campaigns + 1)) * 10
config.speeds = np.random.normal([0.4, 0.4, 0.4, 0.4, 0.4], 0.0)
config.sigmas = np.array([0.5*(i+1) for i in range(config.sub_campaigns)])
config.max_budget = sum(config.speeds) * 5  # half the maximum required to reach the saturation of all the arms
config.n_arms = 10
config.arms = np.linspace(0.0, config.max_budget, config.n_arms).T

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
    subcampaign = Subcampaign(budgets=config.arms,
                              function=lambda x: fun(x, config.alpha_bar[i], config.speeds[i]),
                              sigma=config.sigmas[i])
    env.add_subcampaign(subcampaign)

# for j in range(config.sub_campaigns):
# for i in range(20):
#     plt.scatter(config.arms, env.round(subcampaign=1), s=5)
# plt.show()

x = config.arms.reshape(1 , -1)
y = env.round(subcampaign=1).reshape(1, -1)
learner = GPTS_Learner(config.n_arms, config.arms)
learner.gp.fit(config.arms, env.round(subcampaign=1))
learner.gp.predict(np.atleast_2d(config.arms).T, return_std=True)
x_pred = config.arms
y_pred = learner.means
sigmas = learner.sigmas

def n(k):
    return fun(k, x_bar=config.alpha_bar[j], speed=config.speeds[j])

plt.scatter(x, y, s=5, label=u'Observed Clicks')
plt.plot(x_pred, n(np.array(x_pred)), 'r', label=u'Alpha')
plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
         np.concatenate([y_pred - 1.96 * sigmas, (y_pred + 1.96 * sigmas)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% conf interval')
plt.xlabel('$x$')
plt.ylabel('$n(x)$')
plt.legend(loc="lower right")
plt.show()