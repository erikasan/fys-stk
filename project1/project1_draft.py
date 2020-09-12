import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.utils import resample

class project1:
    def __init__(self, seed=4):
        np.random.seed(seed)
        pass

    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def mean_squared_error(self, y_data, y_model):
        n = np.size(y_model)
        return 1/n*np.sum(y_data - y_model)**2


    def R_squared(self, y_data, y_model):
        return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))**2)

    def design_matrix(self, x, y, degree):
        #Y = np.polynomial.polynomial.polyvander2d(x, y, [1,1])
        N = len(x)
        l = int((degree+1)*(degree+2)/2)
        X = np.ones((N, l))
        for i in range(1, degree+1):
            q = int((i)*(i+1)/2)
            for j in range(i+1):
                X[:,q+j] = x**(i-j)*y**j
        return X

    def OLS(self, X, f):
        #beta = np.linalg.inv(X.T @ X) @ X.T @ f #this is much slower
        beta =np.linalg.pinv(X) @ f
        fpred = X @ beta
        return beta, fpred

    def generate_data(self, n, m, noise=False):
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, m)
        f_data = self.FrankeFunction(x, y)
        if noise == True:
            f_data = f_data + np.random.normal(0, 1)
        return x, y, f_data

    def splitandscale(self, X, f_data):
        X_train, X_test, f_train, f_test = train_test_split(X, f_data, test_size=0.2)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train, X_test, f_train, f_test

    def bootstrap(self, f_data, X, N_bootstrap, degree):
        for degree in range(maxdegree):
            f_predict = np.empty((f_train.shape[0], N_bootstrap))
            f_predict_test = np.empty((f_test.shape[0], N_bootstrap))
            for i in range(N_bootstrap):
                X_train, X_test, f_train, f_test = self.splitandscale(X, f_data)
                X_, f_ = resample(X_train, f_train)

                f_predict[:,i] = self.OLS(X_, f_)[1]
                f_predict_test[:,i] = self.OLS(X_, f_)[1]
        return y_predict



if __name__ == '__main__':
    degrees = np.arange(0, 30, 1)
    MSEtest = np.zeros(len(degrees))
    MSEtrain = np.zeros(len(degrees))

    for i in degrees:
        P1 = project1()
        x, y, f_data = P1.generate_data(100, 100, False)
        X = P1.design_matrix(x, y, degrees[i])

        #not splitted into test and train
        # beta, f_model = P1.OLS(X, f_data)
        # print("Mean Squared Error: ")
        # print(P1.mean_squared_error(f_data, f_model))
        # print("R2 score: ")
        # print(P1.R_squared(f_data, f_model))


        X_train_scaled, X_test_scaled, f_train, f_test = P1.splitandscale(X, f_data)
        beta, f_train_model = P1.OLS(X_train_scaled, f_train)
        f_test_model = X_test_scaled @ beta
        print("Training R2 for OLS")
        print(P1.R_squared(f_train, f_train_model))
        print("Training MSE for OLS")
        print(P1.mean_squared_error(f_train, f_train_model))
        MSEtrain[i] = P1.mean_squared_error(f_train, f_train_model)
        print("Test R2 for OLS")
        print(P1.R_squared(f_test, f_test_model))
        print("Test MSE for OLS")
        print(P1.mean_squared_error(f_test, f_test_model))
        MSEtest[i] = P1.mean_squared_error(f_test, f_test_model)

    print(MSEtest)
    print(MSEtrain)
    plt.plot(degrees, np.log(MSEtest), label='test')
    plt.plot(degrees, np.log(MSEtrain), label='train')
    plt.legend()
    plt.show()




#   beginning of the bootstrap method, not ready yet
#     N_bootstrap = 100
#     y_pred = P1.bootstrap(f_data, X, N_bootstrap, degree)
#
#     # complexity = np.arange(0, 10, 1)
#     # prediction_error = np.zeros(len(complexity))
#     # for i in range(len(complexity)):
#     #     X2 = P1.design_matrix(x, y, i)
#     #     x_train, x_test, f_train2, f_test2 = P1.splitandscale(X2, f_data)
#     #     beta2, f_train_model2 = P1.OLS(x_train, f_train2)
#     #     f_test_model2 = x_test @ beta2
#     #     prediction_error_train[i] = P1.mean_squared_error(f_train2, f_train_model2)
#     #     prediction_error_test[i] = f_data -
#     # print(prediction_error.shape)
#     # print(complexity.shape)
#     # plt.plot(complexity, prediction_error)
#     # plt.show()
#
#
# # plotting the data and the model
#     x1, x2 = np.meshgrid(np.sort(x), np.sort(y))
#     f = P1.FrankeFunction(x1, x2)
#     # plt.imshow(f)
#     # plt.show()
#     fpred = P1.OLS(X, f)[1]
#     fplot = fpred.reshape(100, 100)
#
#     fig = plt.figure()
#
#     ax = fig.add_subplot(1, 2, 1, projection ='3d')
#     surf = ax.plot_surface(x1, x2, f, cmap=cm.viridis, linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     plt.title('Franke')
#
#     ax = fig.add_subplot(1, 2, 2, projection='3d')
#     surf = ax.plot_surface(x1, x2, fplot, cmap=cm.viridis, linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     plt.title('Franke fitted')
#
#     plt.show()


