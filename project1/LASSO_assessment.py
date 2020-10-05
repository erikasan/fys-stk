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

n_datapoints = 1000


x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 5
p_max = 15
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-20, -1, 25)

MSE = np.zeros((lambdas.size, polynomial_degrees.size))

for j, p in enumerate(polynomial_degrees):
    X = design_matrix(x, y, p, pandas = False)
    for i, lam in enumerate(lambdas):
        lasso_reg = Lasso(alpha = lam, tol = 1e-4)
        MSE[i, j] = np.mean(-cross_val_score(lasso_reg, X, z, scoring = 'neg_mean_squared_error'))

vmin = MSE.min()
vmax = MSE.max()
#vmax = 0.006

sns.set()

MSE = pd.DataFrame(MSE)


sns.heatmap(MSE,
            square      = True,
            xticklabels = polynomial_degrees,
            yticklabels = np.round(np.log10(lambdas), 2),
            cmap        = 'rainbow',
            vmin        = vmin,
            vmax        = vmax)

plt.xlabel(r'Polynomial degree')
plt.ylabel(r'$\log( \lambda )$')

plt.show()
