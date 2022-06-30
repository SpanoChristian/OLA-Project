import numpy as np
import math

# Parameters
obs_2nd_slot = 0.4        # Lambda parameter
ratio_users = np.random.dirichlet((1, 10, 1, 1, 1))  # Alpha vector

# Weights matrix
# Rows: secondary products
# Cols: primary products
W = np.array([[0, 0.1, 0.3, 0.2, 0.1], [0.1, 0, 0.2, 0.4, 0.2], [0.2, 0.3, 0, 0.2, 0.1],
              [0.3, 0.3, 0.1, 0, 0.2], [0.2, 0.1, 0.3, 0.2, 0]])

# Display matrix
# This matrix will be updated whenever a product will be displayed as primary
# Initialization: all ones
D = np.ones([5, 5], dtype=int)
D = np.array([[1,1,1,1,1], [0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1], [1,1,1,1,1]])

# Probabilities of observing SLOT 1
print(f"Probabilities of observing SLOT 1: \n {W*D}")

# Probabilities of observing SLOT 2
print(f"Probabilities of observing SLOT 2: \n {obs_2nd_slot*W*D}")

print(ratio_users)

budget = np.array([4, 20, 2, 0.5, 3])
ratio_users = 0.4 * (1 - np.exp(-np.sqrt(budget)))

print(f"\n\nRatio users: {ratio_users}")