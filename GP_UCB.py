import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from Learner import Learner


class GP_UCB_Learner(Learner):
    def __init__(self, n_arms, arms, alpha=0.5):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 0.7
        self.nt = [0.0001 for i in range(0, n_arms)]
        self.beta = [1.0 for i in range(0, n_arms)]

        self.pulled_arms = []
        # default:
        # kernel = RBF(1.0, (1e-5, 1e5))
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # kernel = RBF(1.0, (1e-3, 1e3)) * ConstantKernel(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                                           normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update_betas(self):
        for i in range(0, len(self.arms)):
            self.beta[i] = math.sqrt((2 * math.log(self.t)) / self.nt[i])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        self.nt[pulled_arm] += 1
        self.update_betas()

    def pull_arm(self):
        sampled_values = np.argmax(self.means + self.sigmas * self.beta)
        return sampled_values

    def pull_all_arms(self):
        sampled_values = np.maximum(np.random.normal(self.means, self.sigmas), 0)
        return sampled_values
