import numpy as np
from utils.graph_algorithm import get_graph_paths
from utils.utils import *


class Environment:
    def __init__(self, subcampaign_class):
        pass

    def get_reward(self, subcampaign):
        raise NotImplementedError

    def get_all_clicks(self, subcampaign, clicks):
        raise NotImplementedError

    def compute_rewards(self, pulled_arms):
        raise NotImplementedError

    def round(self, subcampaign=None, pulled_arm=None):
        raise NotImplementedError


class Subcampaign:
    def __init__(self):
        pass

    def update_means(self, alpha_bar, speed):
        raise NotImplementedError

    def round(self, arm_idx=None):
        raise NotImplementedError
