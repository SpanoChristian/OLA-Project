from Environments.Environment_step7 import *
from Learners.GPTS_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils.MKCP import *
from Runner import *

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

environment_aggregate = Environment7(n_subcampaigns=config.n_subcampaigns,
                                     subcampaign_class=Subcampaign4,
                                     alpha_bars=ratios[1:],
                                     multiplier=100000000,
                                     speeds=speeds,
                                     opponent=ratios[0],
                                     adj_matrix=config.adj_matrix,
                                     budgets=np.linspace(0, sum(5 / np.array(speeds)) / 2, 50),
                                     daily_clicks=100
                                     )
runner_aggregate = Runner(environment=env, optimizer=mkcp_solver, learnerClass=GPTS_Learner)

start = time.time()
T = 15
runner.run(T)


class Feature:
    def __init__(self, values, probabilities):
        self.value = None
        self.values = values
        self.probabilities = probabilities


class Context:
    def __init__(self, features, probability):
        self.features = features
        self.probability = probability

    def get_split(self):
        non_assigned_features = [feature for feature in self.features if feature.value is None]

        if len(non_assigned_features) > 0:
            index = features.index(non_assigned_features[0])
            context0_features = features.copy()
            context1_features = features.copy()
            context0_features[index].value = non_assigned_features[0].values[0]
            context1_features[index].value = non_assigned_features[0].values[1]

            return [Context(context0_features, self.probability * context0_features[index].probabilities[0]),
                    Context(context1_features, self.probability * context1_features[index].probabilities[1])]
        else:
            return []


def get_score(context):
    return 1


def get_score_merged(context0, context1):
    return 1


def context_generator(base_context):
    contexts = [base_context]
    final = []
    while len(contexts) > 0:
        context = contexts.pop()
        score_aggregate = get_score(context)
        sub_contexts = context.get_split()
        if len(sub_contexts) > 0:
            score_disaggregate = get_score_merged(sub_contexts[0], sub_contexts[1])
            if score_disaggregate > score_aggregate:
                contexts.extend([sub_contexts[0], sub_contexts[1]])
            else:
                final.append(context)
        else:
            final.append(context)
