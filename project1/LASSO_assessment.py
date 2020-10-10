import numpy as np


from mylearn.ml_tools import *
from FrankeFunction import FrankeFunction

from sklearn.linear_model import Lasso
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


np.random.seed(16091995)

n_datapoints = 1000
bootstraps = 100

x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 2
p_max = 30
polynomial_degrees = np.arange(p_min, p_max + 1, 1)

lambdas = np.logspace(-20, -1, 50)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)



scaler = StandardScaler()


bias      = np.zeros((lambdas.size, polynomial_degrees.size))
variance  = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_cross = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_boot  = np.zeros((lambdas.size, polynomial_degrees.size))

for j, p in enumerate(polynomial_degrees):
    X        = designMatrix(x, y, p, with_intercept = False)
    X_scaled = scaler.fit_transform(X)
    for i, lam in enumerate(lambdas):
        lasso_reg       = Lasso(alpha = lam)#, warm_start = True, precompute = True)
        MSE_cross[i, j] = np.mean(-cross_val_score(lasso_reg, X_scaled, z, scoring = 'neg_mean_squared_error'))

for j, p in enumerate(polynomial_degrees):
    z_predict      = np.zeros((z_test.shape[0], bootstraps))
    X_train        = designMatrix(x_train, y_train, p, with_intercept = False)
    X_test         = designMatrix(x_test, y_test, p, with_intercept = False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.fit_transform(X_test)

    for i, lam in enumerate(lambdas):
        lasso_reg = Lasso(alpha = lam)#, warm_start = True, precompute = True)

        for B in range(bootstraps):
            X_, z_                = resample(X_train_scaled, z_train)
            lasso_reg.fit(X_, z_)
            z_predict[:, B]       = lasso_reg.predict(X_test_scaled)

        bias[i, j]     = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
        variance[i, j] = np.mean(np.var(z_predict, axis=1, keepdims=True))
        MSE_boot[i, j] = mean_squared_error(z_test, np.mean(z_predict, axis=1, keepdims=True))


np.save('bias_LASSO', bias)
np.save('variance_LASSO', variance)
np.save('MSE_boot_LASSO', MSE_boot)
np.save('MSE_cross_LASSO', MSE_cross)

#Plot MSE

# vmin = MSE_cross.min()
# vmax = MSE_cross.max()
# #vmax = 0.006
#
# sns.set()
#
# MSE = pd.DataFrame(MSE_cross)
#
#
# sns.heatmap(MSE_cross,
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
