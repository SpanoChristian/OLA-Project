import numpy as np
from utils.graph_algorithm import get_graph_paths
from Environments.Base_Environment import *
from scipy.stats import t


class Environment4(Base_Environment):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix, budgets,
                 daily_clicks):
        self.multiplier = multiplier
        super().__init__(n_subcampaigns, subcampaign_class, np.array(alpha_bars), speeds, opponent, adj_matrix,
                         budgets, daily_clicks)


    def compute_rewards(self, pulled_arms):
        """
        calls super to compute the partitions, multiply everything by the multiplier (to regulate variance) and
        then applies the dirichlet
        :param pulled_arms: arms pulled
        """
        vals = []
        for i in range(self.n_subcampaigns):
            pulled_arm = pulled_arms[i]
            vals.append(self.round(subcampaign=i, pulled_arm=pulled_arm))
        # assert sum(vals) < 1: conceptually wrong:
        # we have ratios among only primary product and opponent. We should not consider
        # secondary clicks (that are automatically added inside self.round(...)
        k = [1 - sum(vals)]
        k.extend(np.array(vals))
        k = np.array(k) * self.multiplier
        k = (np.random.dirichlet(k) * self.daily_clicks)[1:]
        return [self.get_all_clicks(i, clicks) for i, clicks in enumerate(k)]


class Subcampaign4(Base_Subcampaign):
    def __init__(self, budgets, alpha_bar, speed):
        super().__init__(budgets, alpha_bar, speed)
