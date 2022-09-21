import numpy as np
from Learners.Learner import *
from Environments.Environment import *
from Learners.GPTS_Learner import *

class Runner:
    def __init__(self, environment: Environment, optimizer, learnerClass, dont_update_before=0, **learnerArgs):
        self.environment: Environment = environment
        self.optimizer = optimizer
        self.learnerClass = learnerClass
        self.learnerArgs = learnerArgs
        self.dont_update_before = dont_update_before
        self.learners = []
        self.pulled_super_arms = []

    def run(self, T=40):

        for i in range(0, len(self.environment.alpha_bars)):
            learner: Learner = self.learnerClass(self.environment.budgets, **self.learnerArgs)
            learner.set_horizon(T)
            self.learners.append(learner)

        for i in range(0, T):
            # if i == 35:
            #     for learner in self.learners:
            #         print(["{:.2f}".format(arm) for arm in np.array(learner.pull_all_arms())])
            #     print('\n\n')

            samples = np.zeros(shape=(0, len(self.environment.budgets)))
            for learner in self.learners:
                tmp = np.array(learner.pull_all_arms())
                samples = np.append(samples, [tmp], axis=0)
                self.environment.set_phase(learner.get_phase())

            arms = self.optimizer(samples)
            self.pulled_super_arms.append(arms)
            rewards = self.environment.compute_rewards(arms)


            for j in range(self.environment.n_subcampaigns):
                arm_reward = rewards[j]

                if i < self.dont_update_before:
                    self.learners[j].update_observations(arms[j], arm_reward)
                else:
                    self.learners[j].update(arms[j], arm_reward)

            self.environment.next_day()