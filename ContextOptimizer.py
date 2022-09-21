from Runner import Runner
from Environments.Environment import *
from utils.MKCP import *
import copy

class Feature:
    def __init__(self, feature_value, values):
        self.value = feature_value
        self.values = values

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Context:
    def __init__(self, features, probability, env):
        self.features = features
        self.probability = probability
        self.env = env

    def get_split(self):
        non_assigned_features = [feature for feature in self.features if feature.value is None]

        if len(non_assigned_features) > 0:
            index = self.features.index(non_assigned_features[0])
            context0_features = copy.deepcopy(self.features)
            context1_features = copy.deepcopy(self.features)
            context0_features[index].value = non_assigned_features[0].values[0]
            context1_features[index].value = non_assigned_features[0].values[1]

            return [context0_features, context1_features]
        else:
            return []


class ContextOptimizer:
    def __init__(self, contexts, learnerClass, horizon):
        self.base_context = contexts[0]
        self.contexts = contexts
        self.learnerClass = learnerClass
        self.horizon = horizon

    def get_score(self, context):
        env = context.env
        runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=self.learnerClass)
        runner.run(self.horizon)
        print_experiment(runner)
        rewards = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in range(self.horizon)]
        return compute_best_arm(rewards=rewards, pulled_super_arms=runner.pulled_super_arms)

    def get_combined_score(self, context0, context1):
        env0 = context0.env
        env1 = context1.env
        env = env0.merge(env1)
        runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=self.learnerClass)
        runner.run(self.horizon)
        rewards0 = [sum([learner.collected_rewards[i] for learner in runner.learners[:env0.n_subcampaigns]])
                    for i in range(self.horizon)]
        rewards1 = [sum([learner.collected_rewards[i] for learner in runner.learners[env0.n_subcampaigns+1:]])
                    for i in range(self.horizon)]
        pulled_super_arms0 = np.array(runner.pulled_super_arms)[:, list(range(env0.n_subcampaigns))]
        pulled_super_arms1 = np.array(runner.pulled_super_arms)[:, list(range(env0.n_subcampaigns+1, env1.n_subcampaigns))]

        score_context0, _, _ = compute_best_arm(rewards0, pulled_super_arms0)
        score_context1, _, _ = compute_best_arm(rewards1, pulled_super_arms1)

        return score_context0*context0.probability + score_context1*context1.probability

    def get_context_from_features(self, features):
        return list(filter(lambda context: context.features == features, self.contexts))[0]

    def run(self):
        contexts = [self.base_context]
        final = []
        while len(contexts) > 0:
            context = contexts.pop()
            score_aggregate, _ = self.get_score(context)
            sub_contexts_feature = context.get_split()
            if len(sub_contexts_feature) > 0:
                contex0 = self.get_context_from_features(sub_contexts_feature[0])
                contex1 = self.get_context_from_features(sub_contexts_feature[1])
                score_disaggregate = self.get_combined_score(contex0, contex1)
                if score_disaggregate > score_aggregate:
                    contexts.extend([context0, context1])
                else:
                    final.append(context)
            else:
                final.append(context)

        return final
