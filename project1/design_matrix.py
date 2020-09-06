import numpy as np
from scipy.special import factorial

def design_matrix(x, y, N):
    """
    Returns the design matrix that models
    the output z = f(x, y) as a 2-variable polynomial of degree N.

    x, y: Arrays of input variables (must have same length!)
    N:    Degree of polynomial

    """

    cols = factorial(N+2)/(factorial(N)*factorial(2)) # Number of columns in the design matrix,
    cols = int(cols)                                  # i.e the number of terms in the polynomial
    rows = len(x)                                     # Number of rows in the design matrix

    X = np.zeros((rows, cols))

    col = 0
    for j in range(N+1):
        for i in range(N+1):
            if i + j > N:
                continue
            X[:, col] = x**i * y**j
            col += 1

    return X

"""
# Test implementation on some degree 1 polynomial

def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

x = np.random.rand(3) # Only 3 data points is enough
y = np.random.rand(3) # A first degree polynomial is also the equation for a plane
                      # A plane is uniquely defined by 3 points

X = design_matrix(x, y, 1)

z = np.zeros(len(x))  # Generate a "data set", output z from inputs x and y
z = f(x, y)

beta = np.linalg.inv(X.T @ X) @ X.T @ z
"""

"""
In [10]: beta
Out[10]: array([ 3.14159265,  2.71828183, 42.        ])
"""
