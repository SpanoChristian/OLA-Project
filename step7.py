from Environments.Environment_step7 import *
from Learners.GPTS_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils.MKCP import *
from Runner import *
import json
from ContextOptimizer import *

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


def print_results(runner):
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


def compute_lower_bound(runner):
    pass


## aggregate data
ratios = to_sum_1(np.array([2.5, 2, 3, 1.6, 3, 3.9]))

speeds = [0.1, 0.5, 1.5, 3, 2]
adj_matrix = np.array([
    [0, 0, 0.04, 0.07, 0.9],
    [0, 0, 0.03, 0.03, 0.7],
    [0, 0, 0, 0.2, 0.5],
    [0.02, 0.02, 0, 0, 0],
    [0, 0.01, 0, 0, 0]
])
budgets = np.linspace(0, sum(5 / np.array([0.7, 0.9, 0.5, 0.8, 0.8])) / 0.6, 20)
f = open('./contexts.json')
data = json.load(f)
feature_values = [["m", "f"], ["y", "a"]]
contexts = []
for context in data:
    env = Environment4(n_subcampaigns=5,
                       subcampaign_class=Subcampaign4,
                       alpha_bars=context['ratios'][1:],
                       multiplier=100000,
                       speeds=context['speeds'],
                       opponent=context['ratios'][0],
                       adj_matrix=adj_matrix,
                       budgets=budgets,
                       daily_clicks=100
                       )
    features = [Feature(feature_value=feature, values=feature_values[i])
                for i, feature in enumerate(context["features"])]
    contexts.append(Context(features=features, env=env, probability=context["probability"]))


# print my, ma, fy
def f(alpha_bar, speed, budget):
    return alpha_bar * (1.0 - np.exp(-budget * speed))


x = np.linspace(0, budgets[-1], 50)
selected_contexts = list(filter(
    lambda ctx: list(map(lambda feature: feature.value, ctx.features)) in [["m", "y"], ["f", "y"], ["m", "a"]],
    contexts))
ys = np.array([[f(ctx.env.alpha_bars[i], ctx.env.speeds[i], x) for i in range(5)] for ctx in selected_contexts])

gs = gridspec.GridSpec(1, 3)
plt.figure(figsize=(15, 5))
items = ["computer", "tablet", "phone", "headphones", "charger"]
colors = ["b", "g", "r", "c", "m"]
for i, subcampaigns in enumerate(ys):
    ax = plt.subplot(gs[0, i])
    for j, subcampaign in enumerate(subcampaigns):
        ax.set_title(", ".join(list(map(lambda feature: feature.value, selected_contexts[i].features))))
        plt.plot(x, subcampaign, c=colors[j], label=items[j])
    plt.legend()
plt.show()

# c = ContextOptimizer(contexts=contexts, learnerClass=GPTS_Learner, horizon=20)
# c.run()
