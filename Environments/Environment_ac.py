import numpy as np
from utils.graph_algorithm import get_graph_paths


class Environment:
    def __init__(self, adj_matrix, matrix_sigma, daily_clicks,
                 alpha_bar_low, alpha_bar_high, speed_low, speed_high,
                 opponent_mean, opponent_variance,
                 n_subcampaigns,
                 budgets,
                 phase_tau=40000):
        self.subcampaigns = [Subcampaign(budgets=budgets,
                                         alpha_bar=np.random.uniform(alpha_bar_low, alpha_bar_high),
                                         speed=np.random.uniform(speed_low, speed_high)
                                         ) for i in range(n_subcampaigns)]

        self.adj_matrix = adj_matrix
        self.matrix_sigma = matrix_sigma
        self.reward = [0, 0, 0, 0, 0]
        self.daily_clicks = daily_clicks
        self.t = 0
        self.phase_tau = phase_tau
        self.alpha_bar_low = alpha_bar_low
        self.alpha_bar_high = alpha_bar_high
        self.speed_low = speed_low
        self.speed_high = speed_high
        self.opponent_mean = opponent_mean
        self.opponent_variance = opponent_variance
        self.total = 0
        self.update_total()

    def update_total(self):
        self.total = sum(
            [self.get_all_clicks(i, subcampaign.alpha_bar) for i, subcampaign in enumerate(self.subcampaigns)]) + \
                     np.random.normal(
                         self.opponent_mean, self.opponent_variance)

    def add_subcampaign(self, subcampaign):
        self.subcampaigns.append(subcampaign)

    def next_day(self, arms):
        self.t += 1
        if self.t % self.phase_tau == 0:
            for subcampaign in self.subcampaigns:
                subcampaign.update_means(
                    abs(np.random.uniform(self.alpha_bar_low, self.alpha_bar_high)),
                    abs(np.random.uniform(self.speed_low, self.speed_high))
                )
            self.update_total()

        values_subcampaigns = [self.round(i, arms[i]) for i in range(len(self.subcampaigns))]
        k = [self.total - sum(values_subcampaigns)]
        k.extend(np.array(values_subcampaigns))
        self.reward = (np.random.dirichlet(k) * self.daily_clicks)[1:]

    def get_reward(self, subcampaign):
        return self.reward[subcampaign]

    def get_all_clicks(self, subcampaign, clicks):
        matrix = self.get_matrix()
        paths = get_graph_paths(matrix, subcampaign)
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
    def __init__(self, budgets, alpha_bar, speed):
        self.budgets = budgets
        self.alpha_bar = alpha_bar
        self.speed = speed
        self.means = alpha_bar * (1.0 - np.exp(-budgets * speed))

    def update_means(self, alpha_bar, speed):
        self.alpha_bar = alpha_bar
        self.speed = speed
        self.means = alpha_bar * (1.0 - np.exp(-self.budgets * speed))

    def round(self, arm_idx=None):
        if arm_idx is not None:
            return self.means[arm_idx]
        else:
            return self.means
