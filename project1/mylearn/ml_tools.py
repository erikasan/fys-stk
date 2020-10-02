#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from mylearn.project_tools import fig_path


def rss(ytilde, y):
    """
    RSS - Residual Sum of Squares
    """
    return np.sum((y - ytilde)**2)


def sst(y):
    """
    SST - Sum of Squares Total
    """
    return np.sum((y - np.mean(y))**2)


def r2(ytilde, y):
    """
    Calculate the R^2-score, coefficient of determination (R^2-score)
    """
    return 1 - rss(ytilde, y) / sst(y)


def mse(ytilde, y):
    """
    MSE - Mean Squared Error
    """
    return np.mean((y - ytilde)**2)


def split_data(data, target, test_ratio=0.2):
    shuffled_indices = np.random.permutation(data.shape[0])
    test_set_size = int(data.shape[0] * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices], target[train_indices], target[test_indices]


def frankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) -
                          0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4

# design matrix


def designMatrix(x, y, deg, with_intercept=True):
    """
    Create design matrix

    Set with_intercept to True if an intercept column should be included, False if not

    Note: (not with_intercept) evaluates to 0 if with_intercept is True and 1 if False
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((deg + 1) * (deg + 2) / 2) - \
        (not with_intercept)  # Number of elements in beta
    X = np.ones((N, l))

    idx = 0
    for i in range((not with_intercept), deg + 1):
        for j in range(i + 1):
            X[:, idx] = x**(i - j) * y**j
            idx += 1

    return X


def normalize_data(X_train, X_test, with_intercept=True):
    """
    Normalize training and test dataset w.r.t. training set mean and std.
    If the design matrix is with an intercept column, with_intercept should be
    True, and False if not.
    The intercept column is not normalized.
    """
    if not with_intercept:
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_norm = (
            X_train - X_train_mean[np.newaxis, :]) / X_train_std[np.newaxis, :]
        X_test_norm = (
            X_test - X_train_mean[np.newaxis, :]) / X_train_std[np.newaxis, :]
        return X_train_norm, X_test_norm
    else:
        X_train_mean = np.mean(X_train[:, 1:], axis=0)
        X_train_std = np.std(X_train[:, 1:], axis=0)
        X_train_norm = (
            X_train[:, 1:] - X_train_mean[np.newaxis, :]) / X_train_std[np.newaxis, :]
        X_train_norm = np.c_[np.ones(X_train.shape[0]), X_train_norm]
        X_test_norm = (
            X_test[:, 1:] - X_train_mean[np.newaxis, :]) / X_train_std[np.newaxis, :]
        X_test_norm = np.mylc_[np.ones(X_train.shape[0]), X_test_norm]
        return X_train_norm, X_test_norm


def normalize_X(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return X - X_mean[np.newaxis, :] / X_std[np.newaxis, :]


def plotFranke(save=False):
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = frankeFunction(x, y)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, lw=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title("Franke's function", fontsize=16)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("f(x,y)", fontsize=12, rotation=90)
    ax.view_init(elev=15, azim=45)
    plt.show()
    if save:
        fig.savefig(fig_path("Franke_func.png"), dpi=300)


def plot_CI(CI, save=False):

    labels = []
    for i in range(1, len(CI) + 1):
        labels.append(fr"$\beta_{{{i}}}$")

    fig = plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Reds")
    norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(CI))

    for i in range(len(CI)):
        plt.plot(CI[i], (i, i), color=cmap(norm(i)))
        plt.plot(CI[i], (i, i), "o", color=cmap(norm(i)))

    plt.yticks(np.arange(len(CI)), labels)
    plt.show()
    if save:
        fig.savefig(fig_path("beta_confidence_interval.png"), dpi=300)
