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

    def run(self, T=40):
        for i in range(0, T):
            self.environment.next_day()
            for learnerType in range(self.learnerTypes):
                if i == 35:
                    for learner in self.learners[learnerType]:
                        print(["{:.2f}".format(arm) for arm in np.array(learner.pull_all_arms())])
                    print('\n\n')
                samples = np.zeros(shape=(0, len(self.environment.budgets)))
                for learner in self.learners[learnerType]:
                    tmp = np.array(learner.pull_all_arms())
                    samples = np.append(samples, [tmp], axis=0)
                    self.environment.set_phase(learner.get_phase())

                arms = self.optimizer(samples)
                rewards = self.environment.compute_rewards(arms)


                for j in range(self.environment.n_subcampaigns):
                    arm_reward = rewards[j]

                    if i < self.dont_update_before:
                        self.learners[learnerType][j].update_observations(arms[j], arm_reward)
                    else:
                        self.learners[learnerType][j].update(arms[j], arm_reward)

