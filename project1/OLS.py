import numpy as np


def OLS(X, z):
    """
    Given a design matrix X containing information about the
    input data and the linear model, and the output data z, returns
    the vector beta which minimizes the Euclidean norm |z - X*beta|
    """

    beta = np.linalg.pinv(X) @ z

    return beta
