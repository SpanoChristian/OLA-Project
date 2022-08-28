import logging

import numpy as np
from Environments.Base_Environment import *


class Environment(Base_Environment):
    def __init__(self, n_subcampaigns, alpha_bars, speeds, opponent, adj_matrix, budgets):
        super().__init__(n_subcampaigns, alpha_bars, speeds, opponent, adj_matrix, budgets)
        self.subcampaigns = []

    def add_subcampaign(self, subcampaign):
        self.subcampaigns.append(subcampaign)

    def round(self, subcampaign=None, pulled_arm=None):
        if subcampaign is not None:
            return self.subcampaigns[subcampaign].round(arm_idx=pulled_arm)
        else:
            res = []
            for subcampaign in self.subcampaigns:
                res.append(subcampaign.round(pulled_arm))
            return res


class Subcampaign:
    def __init__(self, budgets, function, sigma):
        self.means = function(budgets)
        self.sigma = sigma

    def round(self, arm_idx=None):
        if arm_idx is not None:
            return np.random.normal(self.means[arm_idx], self.sigma)
        else:
            return np.random.normal(self.means, self.sigma)
