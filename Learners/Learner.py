import numpy as np


class Learner:
    def __init__(self, arms):
        self.n_arms = len(arms)
        self.t = 0
        self.rewards_per_arm = x = [[0] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_model(self):
        raise NotImplementedError

    def update(self, pulled_arm, reward):
        raise NotImplementedError

    def pull_arm(self):
        raise NotImplementedError

    def get_current_t(self):
        pass

    def get_phase(self):
        pass

    def set_horizon(self, horizon):
        pass