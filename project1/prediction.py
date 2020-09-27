import numpy as np

def get_prediction(beta):
    """
    Given the vector of coefficients beta provided by
    OLS, Ridge, LASSO, etc, returns the prediction 'prediction(x, y)'
    that estimates the function 'f(x, y)'.
    """
    cols = beta.shape[0]
    def prediction(x, y):
        z_predict = 0

        N = (np.sqrt(9 + 8*cols) - 3)/2 # Given the number of columns in the design matrix
        N = int(N)                      # N is the polynomial degree
        col = 0
        for j in range(N + 1):
            for i in range(N + 1):
                if i + j > N:
                    continue
                z_predict += x**i * y**j * beta[col]
                col += 1
        return z_predict

    return prediction



# Example of use + verification

"""
import numpy as np
from design_matrix import design_matrix
from ridge import ridge


def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3)
y = np.random.rand(3)
z = f(x, y)

X = design_matrix(x, y, 1, pandas = False)

beta = ridge(X, z, 0)

f_predict = get_prediction(beta)
"""

"""
In [75]: f_predict(0, 0)
Out[75]: 3.141592653589801

In [76]: f_predict(0, 1) - np.pi
Out[76]: 42.0


"""
