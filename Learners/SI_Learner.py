import numpy as np
from copy import copy


def simulate_episode(init_p_matrix, max_iterations):
    p_matrix = init_p_matrix.copy()
    n_nodes = p_matrix.shape[0]
    init_active_nodes = np.random.binomial(1, 0.1, size=n_nodes)
    history = np.array([init_active_nodes])
    active_nodes = init_active_nodes
    new_active_nodes = active_nodes
    t = 0

    while t < max_iterations and np.sum(new_active_nodes) > 0:
        p = (p_matrix.T * active_nodes).T
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        for edge in activated_edges:
            count = 0
            for i, elem in enumerate(edge):
                if count < 2 and elem is True:
                    count += 1
                elif count >= 2 and elem is True:
                    edge[i] = False
        p_matrix *= ((p != 0) == activated_edges)
        new_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        active_nodes = np.array(active_nodes + new_active_nodes)
        history = np.concatenate((history, [new_active_nodes]), axis=0)
        t += 1

    return history


def estimate_probs(data, node_idx, n_nodes):
    credits = np.zeros(n_nodes)
    occurrences_v_active = np.zeros(n_nodes)
    for episode in data:
        idx_w_active = np.argwhere(episode[:, node_idx] == 1).reshape(-1)
        if len(idx_w_active) > 0 and idx_w_active > 0:
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
            credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
        for v in range(0, n_nodes):
            if v != node_idx:
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_v_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
                    occurrences_v_active[v] += 1

    estimated_prob = credits / occurrences_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob


n_episodes = 5000
prob_matrix = np.array([
    [0, 0, 0.04, 0.07, 0.9],
    [0, 0, 0.03, 0.03, 0.7],
    [0, 0, 0, 0.2, 0.5],
    [0.02, 0.02, 0, 0, 0],
    [0, 0.01, 0, 0, 0]
])
n_nodes = len(prob_matrix)
node_idx = 3  # Target node
dataset = []

for e in range(0, n_episodes):
    dataset.append(simulate_episode(init_p_matrix=prob_matrix, max_iterations=10))

estimate_prob = estimate_probs(data=dataset, node_idx=node_idx, n_nodes=n_nodes)

print("True P matrix: ", prob_matrix[:, node_idx])
print("Estimated P matrix: ", estimate_prob)
print("MSE: ", np.mean(np.sum((prob_matrix - estimate_prob) ** 2)))
