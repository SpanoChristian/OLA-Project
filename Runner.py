import numpy as np
from Learners.Learner import *
from Environments.Environment import *


class Runner:
    def __init__(self, environment: Environment, optimizer, lernerClass, **learnerArgs):
        self.environment: Environment = environment
        self.optimizer = optimizer
        self.lernerClass = lernerClass
        self.learnerArgs = learnerArgs
        self.dont_update_before = 1
        self.learners = []

    def run(self, T=40):
        for i in range(0, len(self.environment.alpha_bars)):
            learner: Learner = self.lernerClass(self.environment.budgets, **self.learnerArgs)
            self.learners.append(learner)

        for i in range(0, T):
            samples = np.zeros(shape=(0, len(self.environment.budgets)))
            for learner in self.learners:
                tmp = np.array(learner.pull_all_arms())
                samples = np.append(samples, [tmp], axis=0)

            arms = self.optimizer(samples)
            self.environment.compute_rewards(arms)

            for j in range(self.environment.n_subcampaigns):
                arm_reward = self.environment.get_reward(subcampaign=j)

                if i < self.dont_update_before:
                    self.learners[j].update_observations(arms[j], arm_reward)
                else:
                    self.learners[j].update(arms[j], arm_reward)
