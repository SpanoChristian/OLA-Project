from abc import ABC

import numpy as np
from Environments.Base_Environment import *


class Environment_step6_SW(Base_Environment):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, speeds, opponent, adj_matrix, budgets,
                 daily_clicks, n_arms, probs_matrix, horizon):
        super().__init__(n_subcampaigns, subcampaign_class, np.array(alpha_bars), speeds, opponent, adj_matrix,
                         budgets, daily_clicks)
        self.n_arms = n_arms
        self.probs_matrix = probs_matrix
        self.horizon = horizon
        self.n_changes = len(probs_matrix)
        self.inner_horizon = self.horizon // self.n_changes
        self.t = 0
        self.phase = 0

    def round(self, subcampaign=None, pulled_arm=None):
        if self.t > (self.phase + 1) * self.inner_horizon:
            self.phase = min(self.phase+1, self.n_changes-1)
        # reward * self.probs_matrix[self.phase, pulled_arm]
        self.t += 1
        return reward


class Subcampaign6(Base_Subcampaign):
    def __init__(self, budgets, alpha_bar, speed):
        super().__init__(budgets, alpha_bar, speed)
