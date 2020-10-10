import numpy as np
from sklearn.linear_model import Lasso
from design_matrix import design_matrix

def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

# x = np.random.rand(3)
# y = np.random.rand(3)
x = np.array([0, 1, 0])
y = np.array([0, 0, 1])
z = f(x, y)

X = design_matrix(x, y, 1, pandas = False)

lasso_reg = Lasso(alpha = 0.00001)
lasso_reg.fit(X, z)
lasso_reg.predict(X)
