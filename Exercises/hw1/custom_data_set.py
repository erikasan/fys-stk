# hw1 Excercise 2

import numpy as np



# Generate data set
x = np.random.rand(100)
y = 2*x**2 + np.random.randn(100)


# Subtask 1

# Construct design matrix

n = len(x)
design_matrix = np.vstack((np.ones(n), x, x**2))
design_matrix = design_matrix.T


# Calculate the optimal coefficients
beta = np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y


# Prediction
ypredict = design_matrix @ beta


# Subtask 2

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(design_matrix, y)

beta_scikit   = reg.coef_
ymodel_scikit = design_matrix @ beta_scikit


# Subtask 3

# ZZZZZzzzZZZZZZZzzzzZZZZZzzzz
