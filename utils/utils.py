import numpy as np
from utils.MKCP import *
from scipy.stats import t


def to_sum_1(array: np.ndarray):
    partial = array / np.min(array[np.nonzero(array)])
    return partial / partial.sum()


def compute_cumulative_lower_bounds(rewards, pulled_super_arms):
    lower_bounds = []
    for i in range(T):
        lower_bounds.append([])
        for j in range(np.array(pulled_super_arms).shape[1]):
            pulled = np.array(pulled_super_arms)[:, j]
            mean = np.mean([rewards[k] for k in range(len(pulled)) if pulled[k] == pulled[i]])
            div = 2 * np.count_nonzero(pulled == pulled[i])
            lower_bounds[i].append(mean - np.sqrt(-np.log(0.99999) / div))

    res = [sum(i) for i in lower_bounds]
    _max = max(list(zip(res, runner.pulled_super_arms)), key=lambda x: x[0])
    return _max[0], _max[1]


def get_clairvoyant_score(env, samples):
    pulled_arms = []
    best_rewards = []
    for _ in range(samples):
        optimal_sol = mkcp_solver(env.round())
        rewards = env.compute_rewards(optimal_sol)
        best_rewards.append(sum([rewards[j] for j in range(env.n_subcampaigns)]))
        pulled_arms.append(rewards)
    mean = np.mean(best_rewards)
    sol = pulled_arms[np.argmax([abs(mean - i) for i in best_rewards])]
    std = np.std(best_rewards)
    dof = len(best_rewards) - 1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    confidence_interval = (
        mean - std * t_crit / np.sqrt(len(best_rewards)), mean + std * t_crit / np.sqrt(len(best_rewards)))
    return mean, confidence_interval, sol
