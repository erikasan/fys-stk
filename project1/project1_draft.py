import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.utils import resample

class project1:
    def __init__(self, seed=1):
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
        return 1/n*np.sum((y_data - y_model)**2)

    def R_squared(self, y_data, y_model):
        return 1 - np.sum((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))**2)

    def design_matrix(self, x, y, degree):
        N = len(x)
        l = int((degree+1)*(degree+2)/2)
        X = np.ones((N, l))
        for i in range(1, degree+1):
            q = int((i)*(i+1)/2)
            for j in range(i+1):
                X[:,q+j] = x**(i-j)*y**j
        return X

    def OLS(self, X, f):
        beta = np.linalg.pinv(X) @ f
        return beta

    def beta_var_OLS(self, X, f_data, f_model, p):
        N = len(f_data)
        sigma = 1/(N-p-1)*np.sum((f_data - f_model)**2)
        var_beta = np.linalg.inv(X.T @ X)*sigma**2
        var_beta = np.diag(var_beta)
        return var_beta

    def generate_data(self, n, m, noise):
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, m)
        f_data = self.FrankeFunction(x, y)
        if noise == True:
            f_data_n = f_data + np.random.randn(len(f_data))
        return x, y, f_data

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train[:,1:])
        X_train_scaled = np.ones(X_train.shape)
        X_test_scaled = np.ones(X_test.shape)
        X_train_scaled[:,1:] = scaler.transform(X_train[:, 1:])
        X_test_scaled[:,1:] = scaler.transform(X_test[:,1:])
        return X_train_scaled, X_test_scaled

    def bootstrap(self, datapoints, N_bootstrap, maxdegree, noise):
        bias = np.zeros(maxdegree)
        variance = np.zeros(maxdegree)
        MSE = np.zeros(maxdegree)
        degree = np.arange(1,maxdegree+1)
        x, y, f_data = self.generate_data(datapoints, datapoints, noise)
        for i in range(maxdegree):
            X = self.design_matrix(x, y, degree[i])
            X_train, X_test, f_train, f_test = train_test_split(X, f_data, test_size=0.2)
            X_train, X_test = self.scale(X_train, X_test)
            f_predict = np.empty((f_test.shape[0], N_bootstrap))
            for j in range(N_bootstrap):
                X_, f_ = resample(X_train, f_train)
                f_predict[:,j] = X_test @ self.OLS(X_, f_)
            bias[i] = np.mean((f_test - np.mean(f_predict, axis=1))**2)
            variance[i] = np.mean(np.var(f_predict, axis=1))
            MSE[i] = np.mean(np.mean((f_test - f_predict.T)**2 , axis=1))
        return  bias, variance, MSE

    def kfold(self, datapoints, k, degree, noise):
        #need to scale
        x, y, f_data = self.generate_data(datapoints, datapoints, noise)
        X = self.design_matrix(x, y, degree)
        j = np.arange(len(f_data))
        np.random.shuffle(j)
        split = np.split(j, k)
        MSE = 0
        R2 = 0
        for i in range(k):
            train =np.concatenate((split[:i]+split[i+1:]))
            test = split[i]
            X_train = np.array(X[train])
            f_train = np.array(f_data[train])
            X_test = np.array(X[test])
            f_test = np.array(f_data[test])
            beta = self.OLS(X_train, f_train)
            f_predict = X_test @ beta
            MSE += self.mean_squared_error(f_test, f_predict)
            R2 += self.R_squared(f_test, f_predict)
        return MSE/k, R2/k