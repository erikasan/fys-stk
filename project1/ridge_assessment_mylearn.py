import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mylearn.linear_model import RidgeRegression
from mylearn.ml_tools import *
from FrankeFunction import FrankeFunction

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(16091995)

n_datapoints = 1000
bootstraps = 100


x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 6
p_max = 20
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-20, -1, 100)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

scaler = StandardScaler()


bias      = np.zeros((lambdas.size, polynomial_degrees.size))
variance  = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_boot  = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_cross = np.zeros((lambdas.size, polynomial_degrees.size))


# Cross validation

# for i, lam in enumerate(lambdas):
#     for j, p in enumerate(polynomial_degrees):
#         MSE_cross[i, j] = cross_validation(x, y, z, ridge, p, K = 5, lam = lam)


# Bootstrap

for j, p in enumerate(polynomial_degrees):
    z_predict      = np.zeros((z_test.shape[0], bootstraps))
    X_train        = designMatrix(x_train, y_train, p, with_intercept=False)
    X_test         = designMatrix(x_test, y_test, p, with_intercept=False)
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept=False)

    for i, lam in enumerate(lambdas):
        ridge_reg = RidgeRegression(lmbda=lam, fit_intercept=True)
        for B in range(bootstraps):
            X_, z_                = resample(X_train_scaled, z_train)
            ridge_reg.fit(X_, z_)
            z_predict[:, B]          = ridge_reg.predict(X_test_scaled)
            # beta            = ridge(X_, z_, lam



            # model           = get_prediction(beta)
            #z_predict[:, B] = model(X_test_scaled[:, 1], X_test_scaled[:, p+1])

        bias[i, j]     = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
        variance[i, j] = np.mean(np.var(z_predict, axis=1, keepdims=True))
        MSE_boot[i, j] = mean_squared_error(z_test, np.mean(z_predict, axis=1, keepdims=True))

# Plot MSE

vmin = MSE_boot.min()
vmax = MSE_boot.max()
#vmax = 0.006

MSE_boot = pd.DataFrame(MSE_boot)

sns.set()

sns.heatmap(MSE_boot,
            square      = True,
            xticklabels = polynomial_degrees,
            yticklabels = np.round(np.log10(lambdas), 2),
            cmap        = 'rainbow',
            vmin        = vmin,
            vmax        = vmax)

plt.xlabel(r'Polynomial degree')
plt.ylabel(r'$\log( \lambda )$')

plt.show()

# Plot bias

# vmin = bias.min()
# vmax = bias.max()
#
# bias = pd.DataFrame(bias)
#
# sns.set()
#
# sns.heatmap(bias,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()


# Plot variance

# vmin = variance.min()
# vmax = variance.max()
#
# variance = pd.DataFrame(variance)
#
# sns.set()
#
# sns.heatmap(variance,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()
