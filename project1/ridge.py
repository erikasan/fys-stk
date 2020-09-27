import numpy as np
from design_matrix import design_matrix
from cross_validation import cross_validation

def ridge(X, z, lam):
    """
    Given a design matrix X containing information about the
    input data and the linear model, and the output data z, returns
    the two-variable polynomial function which predicts the output data.


    """

    p = X.shape[1]
    I = np.eye(p)


    beta = np.linalg.inv(X.T@X + lam*I)@X.T@ z

    return beta


# Example of use + Verifaction
"""
def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3)
y = np.random.rand(3)
z = f(x, y)

X = design_matrix(x, y, 1, pandas = False)

lam = 0.00001

beta = ridge(X, z, lam)
print(beta)
"""
"""
In [32]: run ridge.py
[ 3.14925735  2.70942823 41.99388081]
"""
