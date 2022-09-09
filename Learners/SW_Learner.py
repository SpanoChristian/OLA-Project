import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from Learners.Learner import Learner


class SW_Learner(Learner):
    def __init__(self, arms, window_size):
        super().__init__(arms)
        self.window_size = window_size
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 0.7
        self.pulled_arms = []

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        super(SW_Learner, self).update_model()

    def update(self, pulled_arm, reward):
        self.t += 1
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx][-self.window_size:])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf
        self.update_observations(pulled_arm, reward)
        np.append(self.collected_rewards, reward)
        self.rewards_per_arm[pulled_arm].append(reward)

    def pull_arm(self):
        idx = np.argmax(self.means + self.widths)   # UCB
        return idx

    def pull_all_arms(self):
        sampled_values = np.maximum(np.random.normal(self.means, self.sigmas), 0)
        return sampled_values
