from Learner import Learner
import numpy as np


class UCB1(Learner):
    def __init__(self, arms):
        super().__init__(len(arms))
        self.arms = arms
        self.pulled_arms = []

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
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

    def pull_all_arms(self):
        sampled_values = np.maximum(np.random.normal(self.means, self.sigmas), 0)
        return sampled_values
