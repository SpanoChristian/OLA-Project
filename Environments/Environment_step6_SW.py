from abc import ABC

import numpy as np
from Environments.Base_Environment import *


class Environment_step6_SW(Base_Environment):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, speeds, opponent, adj_matrix, budgets,
                 daily_clicks, probs_matrix, horizon=100, n_changes=3):
        super().__init__(n_subcampaigns, subcampaign_class, np.array(alpha_bars), speeds, opponent, adj_matrix,
                         budgets, daily_clicks)
        self.probs_matrix = probs_matrix
        self.t = 0
        self.phase = 0

    def round(self, subcampaign=None, pulled_arm=None):
        # print(f"n changes = {self.n_changes}")
        # print(f"inner_horizon = {self.inner_horizon}")
        if subcampaign is not None:
            # print(f"Phase = {self.phase},  pulled arm = {pulled_arm}")
            # print(f"Prob = {self.probs_matrix[self.phase, pulled_arm]}")
            if pulled_arm is not None:
                res = self.subcampaigns[subcampaign].round(arm_idx=pulled_arm) * self.probs_matrix[
                                                                                    self.phase, pulled_arm]
            else:
                res = self.subcampaigns[subcampaign].round(arm_idx=pulled_arm) * self.probs_matrix[self.phase]
            return res
        else:
            res = []
            for subcampaign in range(len(self.subcampaigns)):
                #print(f"Prob = {self.probs_matrix[self.phase, pulled_arm]}")
                if pulled_arm is not None:
                    res.append(self.round(subcampaign, pulled_arm) * self.probs_matrix[self.phase, pulled_arm])
                else:
                    res.append(self.round(subcampaign, pulled_arm) * self.probs_matrix[self.phase])
            return res

    def update_time(self, t):
        self.t = t

    def set_phase(self, phase):
        self.phase = phase


class Subcampaign6(Base_Subcampaign):
    def __init__(self, budgets, alpha_bar, speed):
        super().__init__(budgets, alpha_bar, speed)
