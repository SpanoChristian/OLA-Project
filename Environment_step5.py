import logging

import numpy as np


class Environment:
    def __init__(self, adj_matrix_primary, adj_matrix_secondary):
        self.subcampaigns = []
        self.adj_matrix_primary = adj_matrix_primary
        self.adj_matrix_secondary = adj_matrix_secondary

    def add_subcampaign(self, subcampaign):
        self.subcampaigns.append(subcampaign)

    def round(self, subcampaign=None, pulled_arm=None):
        if subcampaign is not None:
            res = self.subcampaigns[subcampaign].round(arm_idx=pulled_arm)
            res = [i + get_secondary_clicks(pulled_arm, i) for i in res]
            return res
        else:
            res = []
            for subcampaign in self.subcampaigns:
                res.append(subcampaign.round(pulled_arm))
            return res

    def get_secondary_clicks(self, arm):
        return sum(self.adj_matrix_primary[arm]*i)


class Subcampaign:
    def __init__(self, budgets, function, sigma):
        self.means = function(budgets)
        self.sigma = sigma

    def round(self, arm_idx=None):
        if arm_idx is not None:
            return np.random.normal(self.means[arm_idx], self.sigma)
        else:
            return np.random.normal(self.means, self.sigma)
