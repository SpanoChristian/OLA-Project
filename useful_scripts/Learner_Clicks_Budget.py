import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d


# In theory 100 is our alpha_i_bar
# So we imagine that alpha_i_bar is decided by us
# THIS IS THE FUNCTION THAT CHANGE
# SEE THE GRAPH => IT IS THE "EXTENSION" OF THE ONE VARIABLE
def n_bid_budget(x, y):
    # the real function to estimate
    return (1.0 - np.exp(- (x * y))) * 100


def n(x):
    # the real function to estimate
    return (1.0 - np.exp(- x)) * 100


def generate_observation_bid_budget(x, y, noise_std):
    return n_bid_budget(x, y) + np.random.normal(0, noise_std, size=n_bid_budget(x, y).shape)


def generate_observation_budget(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size=n(x).shape)


alpha_bar = 100
n_obs = 50

bids, budgets = np.meshgrid(np.arange(start=0.0, stop=1.0, step=0.001), np.arange(start=0.0, stop=1.0, step=0.001))

X_obs = np.array([0, 0])
z_obs = np.array([0])  # corresponding (to x_obs, y_obs) observed clicks

# Increasing noise => increasing uncertainty
noise_std = 0.1

X_pred = np.array([0, 0])

# Real function printed

for i in range(0, n_obs):
    # Generate a point based on the value of the bid (chosen randomly)
    # at each time a random point is chosen
    new_bid_obs = np.random.choice(bids.ravel(), 1).reshape(-1)[0]
    new_budget_obs = np.random.choice(budgets.ravel(), 1).reshape(-1)[0]
    new_z_obs = generate_observation_bid_budget(new_bid_obs, new_budget_obs, noise_std)

    vec = np.array([new_bid_obs, new_budget_obs])
    X_obs = np.vstack((X_obs, vec))
    z_obs = np.append(z_obs, new_z_obs)

    X = X_obs
    Z = z_obs.ravel().reshape(-1, 1)

    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, normalize_y=True, n_restarts_optimizer=10)

    gp.fit(X, Z)

    Z_pred, sigma = gp.predict(np.array([bids.ravel(), budgets.ravel()]).T, return_std=True)

print(f"bids size={bids.shape}, budget size={budgets.shape}, Z_pred size={Z_pred.shape}")
n_rows, n_columns = bids.shape
Z_pred = Z_pred.reshape(n_rows, n_columns)
print(f"bids size={bids.shape}, budget size={budgets.shape}, Z_pred size={Z_pred.shape}")
# Let's assume that v = 1 (value per click)
max_budgets = np.amax(1*Z_pred, axis=1)
print(f"Max budget per row: {max_budgets}")
print(Z_pred)

# Real function printed
#z = n_bid_budget(bids, budgets)
#ax = plt.axes(projection='3d')
#ax.plot_surface(bids, budgets, z, cmap='viridis', edgecolor='green')
#plt.show()

# Learned
#ax = plt.axes(projection='3d')
#ax.plot_surface(bids, budgets, Z_pred, cmap='viridis', edgecolor='green')
#plt.show()

x_obs = np.array([])
y_obs = np.array([])

for i in range(0, n_obs):
    # Generate a point based on the value of the bid (chosen randomly)
    # at each time a random point is chosen
    new_x_obs = np.random.choice(max_budgets, 1)
    new_y_obs = generate_observation_budget(new_x_obs, noise_std)

    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)

    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()

    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, normalize_y=True, n_restarts_optimizer=10)

    gp.fit(X, Y)

    x_pred = np.atleast_2d(max_budgets).T
    y_pred, sigma = gp.predict(x_pred, return_std=True)

print(f"X size={X.shape}, Y size={Y.reshape(-1, 1).shape}")
print(f"x_pred size={X.shape}, budgets size={max_budgets.reshape(-1, 1).shape}, y_pred={y_pred.shape}")

plt.plot(x_pred, n(x_pred), 'r:', label=r'$n(x)$')
plt.plot(X.ravel(), Y, 'ro', label=u'Observed Clicks')
plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
         np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% conf interval')
plt.xlabel('$x$')
plt.ylabel('$n(x)$')
plt.legend(loc="lower right")
plt.show()


