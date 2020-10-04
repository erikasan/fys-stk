import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from design_matrix import design_matrix
from ridge import ridge
from OLS import OLS
from cross_validation import cross_validation
from prediction import get_prediction
from FrankeFunction import FrankeFunction

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(42)

n_datapoints = 200
bootstraps   = 50

x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) #+ 0.2*np.random.normal(0, 1, n_datapoints)


z.shape = (x.shape[0], 1)

p_min = 1
p_max = 30
polynomial_degrees = np.arange(p_min, p_max + 1, 1)



MSE      = np.zeros(polynomial_degrees.shape)
bias     = np.zeros(polynomial_degrees.shape)
variance = np.zeros(polynomial_degrees.shape)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size = 0.2)

for p in polynomial_degrees:
    z_predict = np.zeros((z_test.shape[0], bootstraps))
    X = design_matrix(x_train, y_train, p, pandas = False)
    for B in range(bootstraps):
        X_, z_          = resample(X, z_train)
        #beta           = ridge(X_, z_)
        beta            = OLS(X_, z_)
        model           = get_prediction(beta)
        z_predict[:, B] = model(x_test, y_test)

    bias[p-p_min]     = np.mean((z_test - np.mean(z_predict, axis=1, keepdims=True))**2)
    variance[p-p_min] = np.mean(np.var(z_predict, axis=1, keepdims=True))

    #MSE[p-p_min] = np.mean(np.mean((z_test - z_predict.T)**2 , axis=1))
    MSE[p-p_min] = cross_validation(x, y, z, OLS, p, K = 5, lam = None)
    # beta         = OLS(X, z_train)
    # model        = get_prediction(beta)
    # z_predict    = model(x_test, y_test)
    # z_predict.shape = (z_predict.shape[0], 1)
    # MSE[p-p_min] = np.mean((z_test - z_predict)**2)

sns.set()
plt.plot(polynomial_degrees, bias, '-o', label = 'Bias')
plt.plot(polynomial_degrees, variance, '-o', label = 'Variance')
plt.plot(polynomial_degrees, MSE, '-o', label = 'MSE')
#plt.plot(polynomial_degrees, bias + variance, '-o', label = 'Bias + Variance', alpha = 0.7)
plt.yscale('log')
plt.legend()
plt.xlabel('Complexity / Polynomial degree ')
plt.show()
