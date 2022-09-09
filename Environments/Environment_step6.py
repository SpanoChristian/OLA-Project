import numpy as np
from utils.graph_algorithm import get_graph_paths
from Environments.Environment_step5 import *


class Environment6(Environment5):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                 sigma_matrix, budgets,
                 daily_clicks, phase):
        super().__init__(n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                         sigma_matrix, budgets,
                         daily_clicks)
        self.phase = phase
        self.t = 0

    def get_new_subcampaigns(self):
        pass

    def compute_rewards(self, pulled_arms):
        self.t += 1
        if self.t % self.phase == 0:
            self.get_new_subcampaigns()
        super().compute_rewards(pulled_arms)

    def get_all_clicks(self, subcampaign, clicks):
        """
        Update the adj_matrix adding some variance and call base function
        :param subcampaign: starting node
        :param clicks: initial clicks
        :return: total number of clicks given the subcampaign connections
        """
        self.adj_matrix = np.random.normal(self.base_matrix, self.sigma_matrix)
        return super().get_all_clicks(subcampaign, clicks)


class Subcampaign5(Subcampaign4):
    def __init__(self, budgets, alpha_bar, speed):
        super().__init__(budgets, alpha_bar, speed)

    def update_means(self, alpha_bar, speed):
        self.alpha_bar = alpha_bar
        self.speed = speed
        self.means = np.maximum(alpha_bar * (1.0 - np.exp(-self.budgets * speed)), self.min_val)

