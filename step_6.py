import time

from Environments.Environment_step6 import *
from Learners.GPTS_Learner import *
from Learners.GPUCB_Learner import *
import logging
import warnings
import matplotlib.gridspec as gridspec
from utils.Optimization_Algorithm import *
import matplotlib.pyplot as plt
from utils.knapsack import *
import multiprocessing

plt.ion()

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


# alpha function to be learnt
def fun(x, x_bar, speed):
    return x_bar * (1.0 - np.exp(-x * speed))


def _time(function):
    _start = time.time()
    res = function()
    return res, time.time() - _start


logging.debug("setting up parameters")

if __name__ == '__main__':


    class Config(object):
        pass


    config = Config()
    config.sub_campaigns = 5
    config.max_budget = 70
    config.n_arms = 100
    config.arms = np.linspace(0.0, config.max_budget, config.n_arms).T

    config.adj_matrix = np.array([[0, 0, 0.2, 0, 0],
                                  [0.1, 0, 0, 0.3, 0],
                                  [0, 0.2, 0, 0.1, 0],
                                  [0.2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]
                                  ])
    # p2 -> p1, p4
    # p1 -> p3
    #
    config.lambda_param = 0.5
    config.second_secondary = np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])

    env = Environment(adj_matrix=config.adj_matrix, matrix_sigma=0.0002, daily_clicks=1000,
                      alpha_bar_high=150000000, alpha_bar_low=1000000, speed_high=0.9, speed_low=0.1,
                      opponent_mean=12000000, opponent_variance=30, n_subcampaigns=5, budgets=config.arms)

    print("created subcampaigns")

    learners = []
    for i in range(config.sub_campaigns):
        learners.append(GPTS_Learner(arms=config.arms, n_arms=config.n_arms))


    def compute_clairvoyant_reward():
        clairvoyant_solution = knapsack_optimizer(np.array(env.round()))
        k = [np.random.normal(env.opponent_mean, env.opponent_variance)]
        env.next_day(clairvoyant_solution)
        k.extend([env.get_reward(subcampaign) for subcampaign in range(config.sub_campaigns)])
        clairvoyant_reward = sum(k[1:])
        return clairvoyant_reward


    T = 51
    x = [[] for i in range(5)]
    y = [[] for i in range(5)]
    y_clairvoyant = []
    y_pred = [[] for i in range(5)]
    x_pred = config.arms
    sigmas = [[] for i in range(5)]
    differences = [0 for i in range(config.sub_campaigns)]

    start = time.time()
    aux = time.time()
    time_learning = 0
    time_optimizer = 0
    gs = gridspec.GridSpec(1, 5)

    for i in range(0, T):
        if i % 50 == 0:
            print(f'iteration: {i}, time since last: {time.time() - aux}, total time: {time.time() - start}')
            aux = time.time()

        samples = np.zeros(shape=(0, config.n_arms))
        for learner in learners:
            tmp = np.array(learner.pull_all_arms())
            samples = np.append(samples, [tmp], axis=0)

        arms, _time_optimizer = _time(lambda: optimization_algorithm(samples))
        time_optimizer += _time_optimizer

        if i % env.phase_tau == 0:
            y_clairvoyant.append(compute_clairvoyant_reward())
        else:
            y_clairvoyant.append(y_clairvoyant[len(y_clairvoyant) - 1])

        env.next_day(arms)

        part_x = [0 for i in range(config.sub_campaigns)]
        part_y = [0 for i in range(config.sub_campaigns)]


        def learn(args):
            if i > 3:
                args[0].update(args[1], args[2])
            else:
                args[0].update_observations(args[1], args[2])

        start_learning = time.time()
        for j in range(config.sub_campaigns):
            learn([learners[j], arms[j], env.get_reward(j)])
        time_learning += time.time() - start_learning

        if i % 50 == 0:
            print(f'time_opt: {time_optimizer}, time_learning: {time_learning}')

        if (i % 2 == 0 and i > 50) or i % 10 == 0:
            #     plt.figure(figsize=(20, 5))
            #     for j in range(config.sub_campaigns):
            #         y_pred[j] = learners[j].means
            #         sigmas[j] = learners[j].sigmas
            #
            #         plt.subplot(gs[0, j])
            #
            #
            #         def n(k):
            #             e = [np.random.normal(env.opponent_mean, env.opponent_variance)]
            #             e.extend(
            #                 [env.get_all_clicks(j, subcampaign.alpha_bar) for subcampaign in env.subcampaigns])
            #
            #             return fun(k, x_bar=(np.random.dirichlet(e) * env.daily_clicks)[j + 1],
            #                        speed=env.subcampaigns[j].speed)
            #
            #
            #         plt.scatter(x[j], y[j], s=5, label=u'Observed Clicks')
            #         plt.plot(x_pred, n(np.array(x_pred)), 'r', label=u'Alpha')
            #         plt.plot(x_pred, y_pred[j], 'b-', label=u'Predicted Clicks')
            #         plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
            #                  np.concatenate([y_pred[j] - 1.96 * sigmas[j], (y_pred[j] + 1.96 * sigmas[j])[::-1]]),
            #                  alpha=.5, fc='b', ec='None', label='95% conf interval')
            #         plt.xlabel('$x$')
            #         plt.ylabel('$n(x)$')
            #         plt.xlim([0, min(config.max_budget, (5 / env.subcampaigns[j].speed) * 1.5)])
            #         plt.ylim([0, max([max(learner.collected_rewards) for learner in learners])])
            #
            #     plt.show()
            #
            print(i)

    x = range(T)
    y = [sum([learner.collected_rewards[i] for learner in learners]) for i in range(T)]

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    plt.plot(x, y, label=u'GPTS reward')
    plt.plot(x, y_clairvoyant, label=u'Clairvoyant reward')
    plt.text(x=T * 0.75, y=np.average(y_clairvoyant) * 0.05, s=f'total time: {"{:.2f}".format(time.time() - start)}')
    # plt.legend(loc="upper left")
    plt.show()
