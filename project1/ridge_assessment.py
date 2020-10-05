import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from design_matrix import design_matrix
from ridge import ridge
from cross_validation import cross_validation
from prediction import get_prediction
from FrankeFunction import FrankeFunction

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(16091995)

n_datapoints = 1000


x = np.random.rand(n_datapoints)
y = np.random.rand(n_datapoints)
z = FrankeFunction(x, y) + 0.05*np.random.normal(0, 1, n_datapoints)


p_min = 6
p_max = 20
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-16, -5, 20)

MSE = np.zeros((lambdas.size, polynomial_degrees.size))

for i, lam in enumerate(lambdas):
    for j, p in enumerate(polynomial_degrees):
        MSE[i, j] = cross_validation(x, y, z, ridge, p, K = 5, lam = lam)


vmin = MSE.min()
#vmax = MSE.max()
vmax = 0.006

MSE = pd.DataFrame(MSE)

sns.set()

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