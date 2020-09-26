import numpy as np
from scipy.special import factorial
import pandas as pd

def design_matrix(x, y, N, pandas = True):
    """
    Returns the design matrix that models
    the output z = f(x, y) as a 2-variable polynomial of degree N.

    x, y:   Arrays of input variables (must have same length!)
    N:      Degree of polynomial
    pandas: Option to return the design matrix as a DataFrame, returns a numpy array if false

    """

    cols = (N + 2)*(N + 1)/2        # Number of columns in the design matrix,
    cols = int(cols)                # i.e the number of terms in the polynomial
    rows = len(x)                   # Number of rows in the design matrix

    X = np.zeros((rows, cols))

    col = 0
    for j in range(N+1):
        for i in range(N+1):
            if i + j > N:
                continue
            X[:, col] = x**i * y**j
            col += 1

    if pandas:
        X = pd.DataFrame(X)

        column_labels = cols*[]
        for j in range(N+1):
            for i in range(N+1):
                if i + j > N:
                    continue
                if i == j == 0:
                    column_labels.append("1")
                elif j == 0 and i != 0:
                    column_labels.append("x^{}".format(i))
                elif i == 0 and j != 0:
                    column_labels.append("y^{}".format(j))
                else:
                    column_labels.append("x^{}*y^{}".format(i, j))

        X.columns = column_labels

    return X

"""
# Test implementation on some degree 1 polynomial

def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3) # Only 3 data points is enough
y = np.random.rand(3) # A first degree polynomial is also the equation for a plane
                      # A plane is uniquely defined by 3 points

X = design_matrix(x, y, 1)

z = f(x, y) # Generate a "data set", output z from inputs x and y

beta = np.linalg.inv(X.T @ X) @ X.T @ z
"""

"""
In [10]: beta
Out[10]: array([ 3.14159265,  2.71828183, 42.        ])
"""
