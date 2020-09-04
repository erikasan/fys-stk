import numpy as np
from scipy.special import factorial

def construct_design_matrix(x, y, N):
    """
    Returns the design matrix that models
    the output z = f(x, y) as a 2-variable polynomial of degree N.

    x, y: Arrays of input variables
    N: Degree of polynomial

    To display this message in the terminal use help(construct_design_matrix)
    """


    p = factorial(N+2)/(factorial(N)*factorial(2)) # Number of columns in the design matrix,
                                                   # i.e the number of terms in the polynomial
    n = len(x)**2                                  # Number of rows in the design matrix

    X = np.zeros((n, int(p)))

    col = 0
    for j in range(N+1):
        for i in range(N+1):
            if i + j > N:
                continue
            for k in range(len(y)):
                X[k*len(x):(k+1)*len(x), col] = x**i * y[k]**j
            col += 1

    return X

"""
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)



Test implementation on some degree 1 polynomial
def f(x, y):
    return np.pi + np.exp(1)*x + 42*y

X = construct_design_matrix(x, y, 1)

z = np.zeros(len(x)**2)       # Generate a "data set", output z from inputs x and y
for k in range(len(y)):
    z[k*len(x):(k+1)*len(x)] = f(x, y[k])

beta = np.linalg.inv(X.T @ X) @ X.T @ z
"""


"""
In [47]: beta
Out[47]: array([ 3.14159265,  2.71828183, 42.        ])
"""
