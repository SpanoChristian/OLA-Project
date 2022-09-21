import numpy as np
from utils.MKCP import *
from scipy.stats import t
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def to_sum_1(array: np.ndarray):
    partial = array / np.min(array[np.nonzero(array)])
    return partial / partial.sum()


def compute_best_arm(rewards, pulled_super_arms):
    """
    Given the rewards and the super arms pulled during an experiment it computes the lower_bounds in order to get
    a reliable measure to get the best arm
    :param rewards: rewards obtained during the experiment
    :param pulled_super_arms: arms pulled during the experiment
    :return: _max[0] is the cumulative lower bound of the best arm and _max[1] is the best arm
    """
    lower_bounds = []
    for i in range(len(rewards)):
        lower_bounds.append([])
        for j in range(np.array(pulled_super_arms).shape[1]):
            pulled = np.array(pulled_super_arms)[:, j]
            mean = np.mean([rewards[k] for k in range(len(pulled)) if pulled[k] == pulled[i]])
            div = 2 * np.count_nonzero(pulled == pulled[i])
            lower_bounds[i].append(mean - np.sqrt(-np.log(0.99999) / div))

    res = [sum(i) for i in lower_bounds]
    _max = max(list(zip(res, pulled_super_arms)), key=lambda x: x[0])
    return _max[0], _max[1]


def get_clairvoyant_score(env, samples):
    pulled_arms = []
    best_rewards = []
    for _ in range(samples):
        optimal_sol = mkcp_solver(env.round())
        rewards = env.compute_rewards(optimal_sol)
        best_rewards.append(sum([rewards[j] for j in range(env.n_subcampaigns)]))
        pulled_arms.append(optimal_sol)
    mean = np.mean(best_rewards)
    sol = pulled_arms[np.argmax([abs(mean - i) for i in best_rewards])]
    std = np.std(best_rewards)
    dof = len(best_rewards) - 1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    confidence_interval = (
        mean - std * t_crit / np.sqrt(len(best_rewards)), mean + std * t_crit / np.sqrt(len(best_rewards)))
    return mean, confidence_interval, sol


def print_experiment(runner, gs=None, row=0):
    gs_given = True
    if gs is None:
        gs_given = False
        gs = gridspec.GridSpec(1, 2)
        plt.figure(figsize=(13, 5))

    plt.subplot(gs[row, 0])
    clairvoyant_mean, clairvoyant_cb, sol = get_clairvoyant_score(runner.environment, 5)
    y_clairvoyant = np.full(len(runner.pulled_super_arms), clairvoyant_mean)
    y_clairvoyant_cb = np.full((len(runner.pulled_super_arms), 2), clairvoyant_cb)

    x = range(len(runner.pulled_super_arms))
    y = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in
         range(len(runner.pulled_super_arms))]

    plt.plot(x, y, label=u'GPTS reward')
    plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')
    plt.fill_between(x, y_clairvoyant_cb[:, 0], y_clairvoyant_cb[:, 1], alpha=.2, fc='r', ec='None',
                     label='95% conf interval')

    # plt.text(x=len(runner.pulled_super_arms) * 0.75, y=np.max(y_clairvoyant) * 0.9, s=f'total time: {"{:.2f}".format(time.time() - start)}')

    plt.subplot(gs[row, 1])
    regret = []
    for i in range(len(x)):
        diff = y_clairvoyant_cb[i][0] - y[i]
        aux = (diff if diff > 0 else 0) + (regret[-1] if i > 0 else 0)
        regret.append(aux)

    plt.plot(x, regret)
    plt.text(x=len(runner.pulled_super_arms) * 0.6, y=regret[-1] * 0.5,
             s=f'total regret: {"{:.2f}".format(regret[-1])}\n'
               f'total clicks: {"{:.2f}".format(sum(y))}')
    if not gs_given:
        plt.show()
