import numpy as np
from utils.graph_algorithm import get_graph_paths
from Environments.Environment_step4 import *


class Environment5(Environment4):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                 sigma_matrix, budgets,
                 daily_clicks):
        super().__init__(n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                         budgets,
                         daily_clicks)
        self.base_matrix = adj_matrix
        self.sigma_matrix = sigma_matrix

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
