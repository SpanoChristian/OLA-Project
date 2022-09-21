import numpy as np
from Learners.Learner import *
from Environments.Environment import *
from Learners.GPTS_Learner import *


class ComparisonRunner:
    def __init__(self, environment: Environment, optimizer, learners, dont_update_before=3):
        self.environment: Environment = environment
        self.optimizer = optimizer
        self.learners = learners
        self.learnerTypes = len(learners)
        self.dont_update_before = dont_update_before
        self.rewards = [[],[],[] ]

    def run(self, T=40):
        for i in range(0, T):

            for learnerType in range(self.learnerTypes):
                samples = np.zeros(shape=(0, len(self.environment.budgets)))
                for learner in self.learners[learnerType]:
                    tmp = np.array(learner.pull_all_arms())
                    samples = np.append(samples, [tmp], axis=0)

                arms = self.optimizer(samples)
                rewards = self.environment.compute_rewards(arms)
                self.rewards[learnerType].append(sum(rewards))

                for j in range(self.environment.n_subcampaigns):
                    arm_reward = rewards[j]
                    if i < self.dont_update_before:
                        self.learners[learnerType][j].update_observations(arms[j], arm_reward)
                    else:
                        self.learners[learnerType][j].update(arms[j], arm_reward)
            self.environment.next_day()

