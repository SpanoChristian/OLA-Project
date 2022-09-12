import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from Learners.Learner import Learner


class SW_Learner(Learner):
    def __init__(self, arms, alpha=0.9, n_changes=3):
        super().__init__(arms)
        self.window_size = 5
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 0.7
        self.pulled_arms = []
        self.pulled_history = [[] for _ in range(len(arms))]
        self.widths = [np.inf for _ in range(self.n_arms)]
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        # kernel = RBF(1.0, (1e-3, 1e3)) * ConstantKernel(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                                           normalize_y=True, n_restarts_optimizer=9)
        self.phase = 0
        self.n_changes = n_changes
        self.horizon = 100
        self.inner_horizon = self.horizon/self.n_changes

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        print(f"{(self.t / 10)*100}%")
        print(f"{self.t} > {self.phase + 1} * {self.inner_horizon}")
        if self.t > (self.phase + 1) * self.inner_horizon:
            #print(f"{self.t} > {self.phase + 1} * {self.inner_horizon}")
            self.phase = min(self.phase+1, self.n_changes-1)
        self.pulled_history[pulled_arm].append(1)
        # Set to zero all the other arms that have not been played
        self.arms_not_played(pulled_arm)
        for idx_arm in range(self.n_arms):
            if self.t < self.window_size:
                n = sum(self.pulled_history[idx_arm][0:self.t])
            else:
                n = sum(self.pulled_history[idx_arm][self.t-self.window_size:self.t])
            if n > 0:
                self.widths[idx_arm] = np.sqrt(2 * np.log(self.t) / n)
            else:
                self.widths[idx_arm] = np.inf
        self.update_model()
        self.update_observations(pulled_arm, reward)

    def pull_arm(self):
        idx = np.argmax(self.means + self.widths)   # UCB
        return idx

    def pull_all_arms(self):
        sampled_values = np.maximum(np.random.normal(self.means, self.sigmas), 0)
        return sampled_values

    def arms_not_played(self, pulled_arm):
        for arm in range(self.n_arms):
            if arm != pulled_arm:
                self.pulled_history[arm].append(0)

    def get_current_t(self):
        return self.t

    def get_phase(self):
        return self.phase

    def set_horizon(self, horizon):
        self.horizon = horizon
