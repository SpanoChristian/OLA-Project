import numpy as np
from utils.graph_algorithm import get_graph_paths
from Environments.Environment_step5 import *


class Environment6(Environment5):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                 sigma_matrix, budgets,
                 daily_clicks, phase):
        self.t = 0
        self.phase = phase
        self.optimal = []
        self.changed = False

        super().__init__(n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                         sigma_matrix, budgets,
                         daily_clicks)

        self.optimal_sol = mkcp_solver(self.round())
        rewards = super().compute_rewards(self.optimal_sol)
        self.optimal_sol_reward = sum([rewards[j] for j in range(self.n_subcampaigns)])
        self.optimal.append({'sol': self.optimal_sol, 'reward': self.optimal_sol_reward, 'start_from': self.t})

    def get_new_subcampaigns(self):
        new_alpha_bar = np.random.dirichlet([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.opponent = new_alpha_bar[0]
        self.speeds = [np.random.uniform(0.001, 40) for i in range(self.n_subcampaigns)]
        for i, subcampaign in enumerate(self.subcampaigns):
            subcampaign.update_means(new_alpha_bar[i + 1], self.speeds[i])

    def compute_rewards(self, pulled_arms):
        if self.t > 0 and self.t % self.phase == 0 and not self.changed:
            self.changed = True
            self.get_new_subcampaigns()
            self.optimal_sol = mkcp_solver(self.round())
            rewards = super().compute_rewards(self.optimal_sol)
            self.optimal_sol_reward = sum([rewards[j] for j in range(self.n_subcampaigns)])
            self.optimal.append({'sol': self.optimal_sol, 'reward': self.optimal_sol_reward, 'start_from': self.t})

        return super().compute_rewards(pulled_arms)

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
