import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# In theory 100 is our alpha_i_bar
# So we imagine that alpha_i_bar is decided by us
def n(x):
    # the real function to estimate
    return (1.0 - np.exp(-5.0 * x)) * 100
    # return (np.exp(-(1/100)*x)) * np.exp(1) * x


def generate_observation(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size=n(x).shape)


n_obs = 1000
bids = np.linspace(0.0, 1.0, 15)  # linear space from 0 to 1 of 20 vals
x_obs = np.array([])  # served bids
y_obs = np.array([])  # corresponding (to x_obs) observed clicks

# Increasing noise => increasing uncertainty
noise_std = 3

for i in range(0, n_obs):
    # Generate a point based on the value of the bid (chosen randomly)
    # at each time a random point is chosen
    new_x_obs = np.random.choice(bids, 1)
    new_y_obs = generate_observation(new_x_obs, noise_std)

    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)

    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()

    if i % 20 == 0:
        theta = 1.0
        l = 1.0
        kernel = ConstantKernel(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=5 ** 2, normalize_y=True, n_restarts_optimizer=10)

        gp.fit(X, Y)

        x_pred = np.atleast_2d(bids).T
        y_pred, sigma = gp.predict(x_pred, return_std=True)

        plt.ion()
        plt.figure()
        plt.plot(x_pred, n(x_pred), 'r:', label=r'$n(x)$')
        plt.plot(X.ravel(), Y, 'ro', label=u'Observed Clicks')
        plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% conf interval')
        plt.xlabel('$x$')
        plt.ylabel('$n(x)$')
        plt.legend(loc="lower right")
        plt.show(block=True)
