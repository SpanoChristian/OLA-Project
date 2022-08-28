import numpy as np
from utils.graph_algorithm import get_graph_paths


def to_sum_1(array: np.ndarray):
    partial = array / array.min()
    return partial / partial.sum()


class Base_Environment:
    def __init__(self, n_subcampaigns, alpha_bars, speeds, opponent, adj_matrix, budgets):
        """
        Base environment to represent the simplest scenario where everything is well-defined
        :param n_subcampaigns: number of subcampaigns
        :param alpha_bars: max percentage of clicks that each subcampaign can reach.
        The sum needs to be one considering also the opponent
        :param speeds: speed of the alpha function
        :param opponent: min percentage of clicks that the opponent takes
        :param adj_matrix: adjacency matrix of the connections between the subcampaigns
        :param budgets: budgets that will be pulled during the execution
        """
        assert abs(sum(alpha_bars) + opponent - 1) < 0.00001
        self.n_subcampaigns = n_subcampaigns
        self.alpha_bars = alpha_bars
        self.speeds = speeds
        self.opponent = opponent
        self.adj_matrix = adj_matrix
        self.budgets = budgets

        self.subcampaigns = [Base_Subcampaign(budgets, alpha_bar=alpha_bars[i], speed=speeds[i])
                             for i in range(n_subcampaigns)]
        self.t = 0
        self.reward = [0 for i in range(n_subcampaigns)]

    def get_reward(self, subcampaign):
        return self.reward[subcampaign]

    def get_all_clicks(self, subcampaign, clicks):
        """
        Given the matrix of the connections between subcampaigns it computes all possible paths
        inside the graph from the starting node 'subcampaign' and then for every path add to the
        initial clicks the ones returned from that path
        :param subcampaign: starting node
        :param clicks: initial clicks
        :return: total number of clicks given the subcampaign connections
        """
        matrix = self.adj_matrix
        paths = get_graph_paths(matrix, subcampaign)
        res = 0
        for path in paths:
            sub_res = 1
            for i, item in enumerate(path[1:]):
                c = matrix[path[i]][item]
                sub_res *= c
            res += sub_res * clicks
        return res

    def compute_rewards(self, pulled_arms):
        """
        given the pulled arms it computes for each subcampaign the number of clicks, considering also the opponent
        counter contribution
        :param pulled_arms: arms pulled
        """
        vals = []
        for i, subcampaign in enumerate(self.subcampaigns):
            vals.append(self.round(pulled_arms[i]))
        assert sum(vals) < 1
        k = [1 - sum(vals)]
        k.extend(np.array(vals))
        self.reward = (to_sum_1(np.array(k)) * self.daily_clicks)[1:]

    def round(self, subcampaign=None, pulled_arm=None):
        """
        For each subcampaign get the value of corresponding arm played.
        If no arm is given it retrieves the values for all the arms
        If no subcampaign is given it returns the previous result of all subcampaigns
        :param subcampaign:
        :param pulled_arm:
        :return:
        """
        if subcampaign is not None:
            res = np.maximum(self.subcampaigns[subcampaign].round(arm_idx=pulled_arm), 0.0001)
            res = self.get_all_clicks(subcampaign, res)
            return res
        else:
            res = []
            for subcampaign in range(len(self.subcampaigns)):
                res.append(self.round(subcampaign, pulled_arm))
            return res


class Base_Subcampaign:
    def __init__(self, budgets, alpha_bar, speed):
        """
        Class for modelling the behavior of a subcampaign
        :param budgets: values over which will be computed the alpha-function
        :param alpha_bar: max_value reachable (from 0 to 1)
        :param speed: speed of the alpha-function
        """
        assert 0 < alpha_bar < 1
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
