import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from design_matrix import design_matrix


def cross_validation(x, y, z, method, N, K = 5):
    """
    Performs K-fold cross validation of a method

    Args:
        x, y (array):      Input data
        z (array):         Output data
        method (function): The method we want to evaluate (e.g OLS, Ridge)
        N (int):           The polynomial degree of the model
        K (int):           The number of partitions of the input data

    Returns:
        MSE (float):       The mean squared error to assess the method

    """

    kfold = KFold(K)

    MSE = 0
    counter = 0
    for train_index, test_index in kfold.split(x):
        x_train = x[train_index]; x_test = x[test_index]
        y_train = y[train_index]; y_test = y[test_index]

        z_train = z[train_index]; z_test = z[test_index]

        X_train = design_matrix(x_train, y_train, N)

        prediction = method(X_train, z_train)

        z_predict = prediction(x_test, y_test)
        MSE += mean_squared_error(z_test, z_predict)

        counter += 1

    MSE /= counter
    return MSE
