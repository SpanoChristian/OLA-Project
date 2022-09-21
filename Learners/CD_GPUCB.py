import math

import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from Learners.Learner import Learner


class ChangeDetector:
    def __init__(self, M, epsilon, margin):
        self.M = M
        self.t = 0
        self.mean = 0
        self.g_minus = 0
        self.g_plus = 0
        self.epsilon = epsilon
        self.margin = margin

    def update_detector(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.mean += sample / self.M
            return 0
        else:
            self.g_plus = max(0, self.g_plus + (sample - self.mean) - self.epsilon)
            self.g_minus = max(0, self.g_minus - (sample - self.mean) - self.epsilon)
            return self.g_minus > self.margin or self.g_plus > self.margin

    def reset(self):
        self.t = 0
        self.mean = 0
        self.g_minus = 0
        self.g_plus = 0


class CD_GPUCB_Learner(Learner):
    def __init__(self, arms, alpha_gp=0.5, alpha_cd=0.05, M_cd=20, epsilon_cd=0.2, margin_cd=1):
        super().__init__(arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 0.7
        self.nt = [0.0001 for i in range(0, self.n_arms)]
        self.beta = [1.0 for i in range(0, self.n_arms)]

        self.pulled_arms = []
        # default:
        # kernel = RBF(1.0, (1e-5, 1e5))
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # kernel = RBF(1.0, (1e-3, 1e3)) * ConstantKernel(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_gp ** 2,
                                           normalize_y=True, n_restarts_optimizer=9)

        self.alpha_cd = alpha_cd
        self.sa_plus = 0
        self.sa_minus = 0
        self.M_cd = M_cd
        self.detectors = [ChangeDetector(M_cd, epsilon_cd, margin_cd) for _ in arms]
        self.pulled_arms_valid = []
        self.rewards_valid = []
        self.changes = []

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

        if self.detectors[arm_idx].update_detector(reward):
            self.rewards_per_arm[arm_idx] = []
            self.pulled_arms_valid = [arm_idx]
            self.rewards_valid = [reward]
            self.changes.append(self.t)
            self.detectors[arm_idx].reset()
        else:
            self.pulled_arms_valid.append(self.arms[arm_idx])
            self.rewards_valid.append(reward)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms_valid).T
        y = self.rewards_valid
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
        sampled_value = np.argmax(self.means + self.sigmas * self.beta) \
            if np.random.binomial(1, self.alpha_cd) else random.randint(0, self.n_arms - 1)

        return sampled_value

    def pull_all_arms(self):
        """
        When all arms are pulled it randomly increases the value of one arm in order to force the
        selection of that arm
        :return:
        """
        sampled_values = np.maximum(np.random.normal(self.means, self.sigmas), 0)
        if np.random.binomial(1, self.alpha_cd):
            rand = random.randint(0, self.n_arms - 1)
            sampled_values = [np.max(sampled_values) * 1000 if rand == i else item
                              for i, item in enumerate(sampled_values)]
        return sampled_values
