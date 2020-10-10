import numpy as np


from mylearn.linear_model import RidgeRegression
from mylearn.ml_tools import *
from FrankeFunction import FrankeFunction

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


np.random.seed(16091995)

n_datapoints = 1000
bootstraps = 100


x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 20
p_max = 21
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-20, -6, 2)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)




bias      = np.zeros((lambdas.size, polynomial_degrees.size))
variance  = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_boot  = np.zeros((lambdas.size, polynomial_degrees.size))
MSE_cross = np.zeros((lambdas.size, polynomial_degrees.size))





# Bootstrap

for j, p in enumerate(polynomial_degrees):
    z_predict      = np.zeros((z_test.shape[0], bootstraps))
    X_train        = designMatrix(x_train, y_train, p, with_intercept = False)
    X_test         = designMatrix(x_test, y_test, p, with_intercept = False)
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept = False)

    for i, lam in enumerate(lambdas):
        ridge_reg = RidgeRegression(lmbda=lam, fit_intercept=True)
        for B in range(bootstraps):
            X_, z_                = resample(X_train_scaled, z_train)
            ridge_reg.fit(X_, z_)
            z_predict[:, B]          = ridge_reg.predict(X_test_scaled)

        bias[i, j]     = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
        variance[i, j] = np.mean(np.var(z_predict, axis=1, keepdims=True))
        MSE_boot[i, j] = mean_squared_error(z_test, np.mean(z_predict, axis=1, keepdims=True))



kfold = KFold(n_splits = 5)
for j, p in enumerate(polynomial_degrees):
    X = designMatrix(x, y, p, with_intercept = False)
    for i, lam in enumerate(lambdas):
        ridge_reg = RidgeRegression(lmbda=lam, fit_intercept=True)
        counter = 0
        for train_inds, test_inds in kfold.split(x):
            X_train = X[train_inds]
            z_train = z[train_inds]

            X_test  = X[test_inds]
            z_test  = z[test_inds]

            X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, with_intercept = False)
            z_train_scaled, z_test_scaled = normalize_target(z_train, z_test)

            ridge_reg.fit(X_train_scaled, z_train_scaled)
            z_predict        = ridge_reg.predict(X_test_scaled)
            MSE_cross[i, j] += mean_squared_error(z_predict, z_test_scaled)

            counter += 1

        MSE_cross[i, j] /= counter

# np.save('bias_ridge', bias)
# np.save('variance_ridge', variance)
# np.save('MSE_boot_ridge', MSE_boot)
# np.save('MSE_cross_ridge', MSE_cross)
