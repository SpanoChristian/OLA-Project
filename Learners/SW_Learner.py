import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from Learners.Learner import Learner
import random

class SW_Learner(Learner):
    def __init__(self, arms, alpha=0.5):
        super().__init__(arms)
        self.window_size = 60
        self.arms = arms
        self.pulled_history = [[] for _ in range(len(arms))]
        self.widths = [10.0 for _ in range(self.n_arms)]

        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 0.7
        self.pulled_arms = []
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        # kernel = RBF(1.0, (1e-3, 1e3)) * ConstantKernel(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                                           normalize_y=True, n_restarts_optimizer=9)
        self.valid_pulled_arms = []
        self.valid_rewards = []

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.valid_pulled_arms).T
        y = self.valid_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update_widths(self):
        for i in range(len(self.arms)):
            n = sum(self.pulled_history[i])
            self.widths[i] = math.sqrt((2 * math.log(self.t)) / (n if n > 0 else 0.0001))

    def update(self, pulled_arm, reward):
        self.t += 1
        self.pulled_history[pulled_arm].append(1)
        # Set to zero all the other arms that have not been played
        self.arms_not_played(pulled_arm)
        self.valid_pulled_arms.append(self.arms[pulled_arm])
        self.valid_rewards.append(reward)
        if self.t > self.window_size:
            self.remove_first_from_history()
            self.valid_pulled_arms.pop(0)
            self.valid_rewards.pop(0)
        self.update_widths()

        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        idx = np.argmax(self.means + self.widths)  # UCB
        return idx

    def pull_all_arms(self):
        sampled_values = np.maximum(self.means + self.sigmas * self.widths, 0)

        if np.random.binomial(1, 0.02):
            rand = random.randint(0, self.n_arms - 1)
            sampled_values = [np.max(sampled_values) * 1000 if rand == i else item
                              for i, item in enumerate(sampled_values)]

        return sampled_values

    def arms_not_played(self, pulled_arm):
        for arm in range(self.n_arms):
            if arm != pulled_arm:
                self.pulled_history[arm].append(0)

    def remove_first_from_history(self):
        for arm in range(self.n_arms):
            self.pulled_history[arm].pop(0)

    def get_current_t(self):
        return self.t

    def get_phase(self):
        return self.phase

    def set_horizon(self, horizon):
        self.horizon = horizon
