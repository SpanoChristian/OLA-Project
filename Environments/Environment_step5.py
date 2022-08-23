import numpy as np
from utils.graph_algorithm import get_graph_paths


class Environment:
    def __init__(self, adj_matrix, matrix_sigma, daily_clicks):
        self.subcampaigns = []
        self.adj_matrix = adj_matrix
        self.matrix_sigma = matrix_sigma
        self.reward = [0, 0, 0, 0, 0]
        self.daily_clicks = daily_clicks

    def add_subcampaign(self, subcampaign):
        self.subcampaigns.append(subcampaign)

    def next_day(self, arms):
        k = [np.random.normal(10000, 1)]
        k.extend(np.array([self.round(i, arms[i]) for i in range(len(self.subcampaigns))])*100)
        self.reward = np.random.dirichlet(k)[1:] * self.daily_clicks

    def get_reward(self, subcampaign):
        return self.reward[subcampaign]

    def get_all_clicks(self, arm, clicks):
        matrix = self.get_matrix()
        paths = get_graph_paths(matrix, arm)
        res = 0
        for path in paths:
            sub_res = 1
            for i, item in enumerate(path[1:]):
                c = matrix[path[i]][item]
                sub_res *= c
            res += sub_res * clicks
        return res

    def get_matrix(self):
        return np.random.normal(self.adj_matrix, self.matrix_sigma)

    def round(self, subcampaign=None, pulled_arm=None):
        if subcampaign is not None:
            res = np.maximum(self.subcampaigns[subcampaign].round(arm_idx=pulled_arm), 0.0001)
            res = self.get_all_clicks(subcampaign, res)
            return res
        else:
            res = []
            for subcampaign in range(len(self.subcampaigns)):
                res.append(self.round(subcampaign, pulled_arm))
            return res


class Subcampaign:
    def __init__(self, budgets, function):
        self.means = function(budgets)

    def round(self, arm_idx=None):
        if arm_idx is not None:
            return self.means[arm_idx]
        else:
            return self.means
