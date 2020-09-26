import numpy as np
from design_matrix import design_matrix



def OLS(X, z):
    """
    Given a design matrix X containing information about the
    input data and the linear model, and the output data z, returns
    the two-variable polynomial function which predicts the output data

    The coefficients of the two-variable function are generated using
    Ordinary Least Squares (OLS)
    """

    beta = np.linalg.pinv(X) @ z
    cols = X.shape[1]

    def prediction(x, y):
        zpredict = 0

        N = (np.sqrt(9 + 8*cols) - 3)/2 # Given the number of columns in the design matrix
        N = int(N)                      # N is the polynomial degree
        col = 0
        for j in range(N+1):
            for i in range(N+1):
                if i + j > N:
                    continue
                zpredict += x**i * y**j * beta[col]
                col += 1
        return zpredict

    return prediction

# Example of use + validation

"""
def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3)
y = np.random.rand(3)
z = f(x, y)

X = design_matrix(x, y, 1, pandas = False)

fpredict = OLS(X, z)
"""

"""
In [46]: fpredict(0, 0) - np.pi
Out[46]: -1.0658141036401503e-14

In [47]: fpredict(1, 0) - np.pi - np.exp(1)
Out[47]: -7.061018436615996e-14

In [48]: fpredict(0, 1) - np.pi - 42
Out[48]: -4.263256414560601e-14
"""
