import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





p_min = 2
p_max = 30
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-20, -1, 50)

xticks = np.arange(2, 29, 2)
yticks = np.arange(2, 50, 10)



# Plot MSE_cross


# MSE_cross = np.load('MSE_cross_LASSO.npy')
#
# vmin = MSE_cross.min()
# vmax = MSE_cross.max()
# vmax = 0.006
#
# MSE_cross = pd.DataFrame(MSE_cross)
#
# sns.set()
#
#
# sns.heatmap(MSE_cross,
#             square      = False,
#             #xticklabels = polynomial_degrees,
#             #yticklabels = np.round(np.log10(lambdas), 2),
#             #xticklabels = xticklabels,
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
#
# plt.title('MSE with 5-fold cross validation')
# plt.xticks(ticks = xticks, labels = xticks, rotation='horizontal')
# plt.yticks(ticks = yticks, labels = np.round(np.log10(lambdas), 0)[0:-1:10], rotation='horizontal')
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
# plt.tight_layout()
# plt.savefig('MSE_cross_LASSO.png', type = 'png')
# plt.show()


# Plot MSE_boot


# MSE_boot = np.load('MSE_boot_LASSO.npy')
#
# vmin = MSE_boot.min()
# vmax = MSE_boot.max()
# vmax = 0.005
#
# MSE_boot = pd.DataFrame(MSE_boot)
#
# sns.set()
#
# sns.heatmap(MSE_boot,
#             square      = False,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.title('MSE with Bootstrap')
# plt.xticks(ticks = xticks, labels = xticks, rotation='horizontal')
# plt.yticks(ticks = yticks, labels = np.round(np.log10(lambdas), 0)[0:-1:10], rotation='horizontal')
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
# plt.tight_layout()
# plt.savefig('MSE_boot_LASSO.png', type = 'png')
# plt.show()



# Plot bias


# bias = np.load('bias_LASSO.npy')
#
#
# vmin = bias.min()
# vmax = bias.max()
#
# bias = pd.DataFrame(bias)
#
# sns.set()
#
# sns.heatmap(bias,
#             square      = False,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.title('Bias with Bootstrap')
# plt.xticks(ticks = xticks, labels = xticks, rotation='horizontal')
# plt.yticks(ticks = yticks, labels = np.round(np.log10(lambdas), 0)[0:-1:10], rotation='horizontal')
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
# plt.tight_layout()
# plt.savefig('bias_LASSO.png', type = 'png')
# plt.show()


# Plot variance


# variance = np.load('variance_LASSO.npy')
#
#
# vmin = variance.min()
# vmax = variance.max()
# #vmax = 0.006
#
# variance = pd.DataFrame(variance)
#
# sns.set()
#
# sns.heatmap(variance,
#             square      = False,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.title('Variance with Bootstrap')
# plt.xticks(ticks = xticks, labels = xticks, rotation='horizontal')
# plt.yticks(ticks = yticks, labels = np.round(np.log10(lambdas), 0)[0:-1:10], rotation='horizontal')
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
# plt.tight_layout()
# plt.savefig('var_LASSO.png', type = 'png')
# plt.show()
