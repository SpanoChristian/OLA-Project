
import numpy as np
from utils.graph_algorithm import get_graph_paths
from Environments.Environment_step4 import *


class Environment7(Environment4):
    def __init__(self, n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                 budgets, daily_clicks):
        super().__init__(n_subcampaigns, subcampaign_class, alpha_bars, multiplier, speeds, opponent, adj_matrix,
                         budgets,  daily_clicks)




