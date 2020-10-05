import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from design_matrix import design_matrix
from FrankeFunction import FrankeFunction

from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


np.random.seed(16091995)

n_datapoints = 10
bootstraps = 3

x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 5
p_max = 10
polynomial_degrees = np.arange(p_min, p_max + 1, 1)

lambdas = np.logspace(-6, -8, 10)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

bias     = np.zeros((lambdas.size, polynomial_degrees.size))
variance = np.zeros((lambdas.size, polynomial_degrees.size))
MSE      = np.zeros((lambdas.size, polynomial_degrees.size))

# for j, p in enumerate(polynomial_degrees):
#     X = design_matrix(x, y, p, pandas = False)
#     for i, lam in enumerate(lambdas):
#         lasso_reg = Lasso(alpha = lam)
#         MSE[i, j] = np.mean(-cross_val_score(lasso_reg, X, z, scoring = 'neg_mean_squared_error'))

for j, p in enumerate(polynomial_degrees):
    z_predict = np.zeros((z_test.shape[0], bootstraps))
    X_train   = design_matrix(x_train, y_train, p, pandas = False)
    X_test    = design_matrix(x_test, y_test, p, pandas = False)

    for i, lam in enumerate(lambdas):
        lasso_reg = Lasso(alpha = lam)

        for B in range(bootstraps):
            X_, z_                = resample(X_train, z_train)
            lasso_reg.fit(X_, z_)
            z_predict[:, B]       = lasso_reg.predict(X_test)

        bias[i, j]     = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
        variance[i, j] = np.mean(np.var(z_predict, axis=1, keepdims=True))

# Plot MSE

# vmin = MSE.min()
# vmax = MSE.max()
# #vmax = 0.006
#
# sns.set()
#
# MSE = pd.DataFrame(MSE)
#
#
# sns.heatmap(MSE,
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


# Plot bias

vmin = bias.min()
vmax = bias.max()

bias = pd.DataFrame(bias)

sns.set()

sns.heatmap(bias,
            square      = True,
            xticklabels = polynomial_degrees,
            yticklabels = np.round(np.log10(lambdas), 2),
            cmap        = 'rainbow',
            vmin        = vmin,
            vmax        = vmax)

plt.xlabel(r'Polynomial degree')
plt.ylabel(r'$\log( \lambda )$')

plt.show()

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
