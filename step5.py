from Environments.Environment_step5 import *
from Environments.Base_Environment import *
from Learners.GPTS_Learner import *
from Learners.GPUCB_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from utils.MKCP import *
from utils.utils import *
from Runner import *

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


class Config(object):
    pass


config = Config()
config.n_subcampaigns = 5
config.ratios = to_sum_1(np.array([3, 2, 3, 4, 5, 6]))
config.alpha_bars = config.ratios[1:]
config.speeds = [np.random.uniform(0.1, 0.9) for i in range(config.n_subcampaigns)]
config.opponent = config.ratios[0]
config.adj_matrix = np.array([
    [0, 0, 0.04, 0.07, 0.9],
    [0, 0, 0.03, 0.03, 0.7],
    [0, 0, 0, 0.2, 0.5],
    [0.02, 0.02, 0, 0, 0],
    [0, 0.01, 0, 0, 0]
])
config.budgets = np.linspace(0, sum(5 / np.array(config.speeds)) / 2, 20)

while True:
    env = Base_Environment(n_subcampaigns=config.n_subcampaigns,
                           subcampaign_class=Base_Subcampaign,
                           alpha_bars=config.alpha_bars,
                           speeds=config.speeds,
                           opponent=config.opponent,
                           adj_matrix=config.adj_matrix,
                           budgets=config.budgets,
                           daily_clicks=100
                           )

    runner = Runner(environment=env, optimizer=mkcp_solver, learnerClass=GPTS_Learner)

    start = time.time()
    T = 30
    runner.run(T)
    print_experiment(runner)

    rewards = [sum([learner.collected_rewards[i] for learner in runner.learners]) for i in range(T)]
    cumulative_lower_bound, computed_best_arm = compute_best_arm(rewards=rewards,
                                                                 pulled_super_arms=runner.pulled_super_arms)
    clairvoyant_mean, clairvoyant_cb, sol = get_clairvoyant_score(runner.environment, 5)
    print(f'best arm computed: {computed_best_arm}, cumulative_lb: {cumulative_lower_bound}, '
          f'reward: {rewards[runner.pulled_super_arms.index(computed_best_arm)]}')
    print(
        f'highest score reached: {max(rewards)}, highest score arm: {runner.pulled_super_arms[rewards.index(max(rewards))]}')
    print(f'optimal arm: {sol}, optimal arm reward: {clairvoyant_mean}')

    best_arm_computed = computed_best_arm
    highest_score_arm = runner.pulled_super_arms[rewards.index(max(rewards))]
    clairvoyant = sol
    records = [[], [], []]

    for i in range(100):
        records[0].append(sum(env.compute_rewards(best_arm_computed)))
        records[1].append(sum(env.compute_rewards(highest_score_arm)))
        records[2].append(sum(env.compute_rewards(clairvoyant)))

    scores = [np.mean(i) for i in records]
    print(scores)
    records = [[], [], []]

    for i in range(100):
        records[0].append(sum(env.compute_rewards(best_arm_computed)))
        records[1].append(sum(env.compute_rewards(highest_score_arm)))
        records[2].append(sum(env.compute_rewards(clairvoyant)))

    scores = [np.mean(i) for i in records]
    print(scores)
    records = [[], [], []]
    for j in range(3):
        env = Base_Environment(n_subcampaigns=config.n_subcampaigns,
                               subcampaign_class=Base_Subcampaign,
                               alpha_bars=config.alpha_bars,
                               speeds=config.speeds,
                               opponent=config.opponent,
                               adj_matrix=config.adj_matrix,
                               budgets=config.budgets,
                               daily_clicks=100
                               )
        for i in range(100):
            if j == 0:
                records[0].append(sum(env.compute_rewards(best_arm_computed)))
            if j == 1:
                records[1].append(sum(env.compute_rewards(highest_score_arm)))
            if j == 2:
                records[2].append(sum(env.compute_rewards(clairvoyant)))
    scores = [np.mean(i) for i in records]
    print(scores)
    records = [[], [], []]

# cumulative_lower_bound, computed_best_arm = compute_best_arm(rewards=y, pulled_super_arms=runner.pulled_super_arms)
# print(f'best arm computed: {computed_best_arm}, cumulative_lb: {cumulative_lower_bound}, '
#       f'reward: {y[runner.pulled_super_arms.index(computed_best_arm)]}')
# print(f'highest score reached: {max(y)}, highest score arm: {runner.pulled_super_arms[y.index(max(y))]}')
# print(f'optimal arm: {sol}, optimal arm reward: {clairvoyant_mean}')
#
# best_arm_computed = computed_best_arm
# highest_score_arm = runner.pulled_super_arms[y.index(max(y))]
# clairvoyant = sol
# records = [[], [], []]
#
# for i in range(100):
#     records[0].append(sum(env.compute_rewards(best_arm_computed)))
#     records[1].append(sum(env.compute_rewards(highest_score_arm)))
#     records[2].append(sum(env.compute_rewards(clairvoyant)))
#
# scores = [np.mean(i) for i in records]
# print(scores)
