import numpy as np


class Environment:
    def __init__(self, graph_weights, budget):
        self.graph_weights = graph_weights
        self.budget = budget
        self.prices = np.array([1.99, 10.99, 3.99, .00, 300.00])   # = margin
        self.categories = {0: "", 1: "", 2: ""}

    def update_budget(self, new_budget):
        self.budget = new_budget
