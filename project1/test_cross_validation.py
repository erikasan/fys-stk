import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from OLS import OLS
from cross_validation import cross_validation
from design_matrix import design_matrix
from prediction import get_prediction
from FrankeFunction import FrankeFunction

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(1000)
y = np.random.rand(1000)
z = FrankeFunction(x, y)

reg = LinearRegression()

p_min = 1
p_max = 30
polynomial_degrees = np.arange(p_min, p_max + 1, 1)

MSE   = np.zeros(polynomial_degrees.shape)
score = np.zeros(polynomial_degrees.shape)

for p in polynomial_degrees:
    MSE[p-p_min]   = cross_validation(x, y, z, OLS, p, K = 5)

    X              = design_matrix(x, y, p, pandas = False)
    score[p-p_min] = np.mean(-cross_val_score(reg, X, z, scoring = 'neg_mean_squared_error'))


sns.set()
plt.plot(polynomial_degrees, MSE, '-o', label = 'My CV')
plt.plot(polynomial_degrees, score, '-o', label = 'sklearn CV')
plt.legend()
plt.show()
