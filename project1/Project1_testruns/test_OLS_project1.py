import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../Project1_codes/project1_draft/')
from ..Project1_codes.project1_draft import Project1
P1 = project1()

def calcOLS():
    degree = 5
    x, y, f_data = P1.generate_data(500, 500, False)
    X = P1.design_matrix(x, y, degree)
    X_train, X_test, f_train, f_test = train_test_split(X, f_data, test_size=0.2)
    X_train, X_test = P1.scale(X_train, X_test)
    beta = P1.OLS(X_train, f_train)
    f_model = X_test @ beta
    print("Mean Squared Error: ")
    print(P1.mean_squared_error(f_test, f_model))
    print("R2 score: ")
    print(P1.R_squared(f_test, f_model))
    beta_var = P1.beta_var_OLS(X_test, f_test, f_model, degree)
    print("Beta variance: ")
    print(beta_var)

calcOLS()

def plotting_test_vs_train():
    P1 = project1()
    x, y, f_data = P1.generate_data(500, 500, True)
    maxdegree = 30
    degrees = np.arange(1, maxdegree)
    MSEtest = np.zeros(len(degrees))
    MSEtrain = np.zeros(len(degrees))
    for i in range(maxdegree-1):
        X = P1.design_matrix(x, y, degrees[i])
        X_train, X_test, f_train, f_test = train_test_split(X, f_data, test_size=0.2)
        X_train_scaled, X_test_scaled = P1.scale(X_train, X_test)
        beta = P1.OLS(X_train_scaled, f_train)
        f_test_model = X_test_scaled @ beta
        f_train_model = X_train_scaled @ beta
        MSEtrain[i] = P1.mean_squared_error(f_train, f_train_model)
        MSEtest[i] = P1.mean_squared_error(f_test, f_test_model)
    plt.plot(degrees, MSEtest, label='test')
    plt.plot(degrees, MSEtrain, label='train')
    plt.title('Test vs Train Error')
    plt.ylabel('MSE')
    plt.xlabel('Complexity (plynomial degree)')
    plt.legend()
    plt.yscale('log')
    plt.show()
plotting_test_vs_train()

def plotting_bootstrap():
    P1 = project1()
    maxdegree = 25
    n_boot = 20
    datapoints = 5000
    bias, variance, MSE = P1.bootstrap(datapoints, n_boot, maxdegree, noise=True)
    polydegree = np.arange(maxdegree)
    plt.plot(polydegree, bias, label='Bias')
    plt.plot(polydegree, variance, label='var')
    plt.plot(polydegree, MSE, label='MSE')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Complexity (Polynomial degree)')
    plt.title('Bias-Variance using bootstrap')
    plt.show()
plotting_bootstrap()

def plotting_kfold():
    P1 = project1()
    degrees = np.arange(25)
    R2 = np.zeros(len(degrees))
    MSE = np.zeros(len(degrees))
    k = 5
    for degree in degrees:
        MSE[degree], R2[degree] = P1.kfold(500, k, degree, False)
    plt.plot(degrees, MSE)
    plt.show()

plotting_kfold()


def plotting_function():
    x1, x2 = np.meshgrid(np.sort(x), np.sort(y))
    f = P1.FrankeFunction(x1, x2)
    fpred = P1.OLS(X, f)[1]
    fplot = fpred.reshape(100, 100)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection ='3d')
    surf = ax.plot_surface(x1, x2, f, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Franke')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x1, x2, fplot, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Franke fitted')
    plt.show()
