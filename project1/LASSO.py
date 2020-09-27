from sklearn.linear_model import Lasso

def LASSO(X, z, lam = 0):

    lasso_reg = Lasso(alpha = lam, fit_intercept = False)
    lasso_reg.fit(X, z)
    beta = lasso_reg.coef_
    return beta

# Example of use + Verifaction

import numpy as np
from design_matrix import design_matrix
from prediction import get_prediction

def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3)
y = np.random.rand(3)
z = f(x, y)

X = design_matrix(x, y, 1, pandas = False)

beta = LASSO(X, z, 0.000001)

f_predict = get_prediction(beta)

"""
In [123]: f_predict(0, 0)
Out[123]: 3.504453997051268

In [124]: f_predict(0, 1) - np.pi
Out[124]: 42.46292884371249

# Guess this is alright?
"""
