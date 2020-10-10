import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





p_min = 2
p_max = 30
polynomial_degrees = np.arange(p_min, p_max + 1, 1)


lambdas = np.logspace(-20, -1, 50)

# Plot MSE_cross


# MSE_cross = np.load('MSE_cross_ridge.npy')

# vmin = MSE_cross.min()
# vmax = MSE_cross.max()
# #vmax = 0.006
#
# MSE_cross = pd.DataFrame(MSE_cross)
#
# sns.set()
#
# sns.heatmap(MSE_cross,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()


# Plot MSE_boot


# MSE_boot = np.load('MSE_boot_ridge.npy')

# vmin = MSE_boot.min()
# vmax = MSE_boot.max()
# #vmax = 0.006
#
# MSE_boot = pd.DataFrame(MSE_boot)
#
# sns.set()
#
# sns.heatmap(MSE_boot,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()



# Plot bias


# bias = np.load('bias_ridge.npy')


# vmin = bias.min()
# vmax = bias.max()
#
# bias = pd.DataFrame(bias)
#
# sns.set()
#
# sns.heatmap(bias,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()


# Plot variance


# variance = np.load('variance_ridge.npy')


# vmin = variance.min()
# vmax = variance.max()
#
# variance = pd.DataFrame(variance)
#
# sns.set()
#
# sns.heatmap(variance,
#             square      = True,
#             xticklabels = polynomial_degrees,
#             yticklabels = np.round(np.log10(lambdas), 2),
#             cmap        = 'rainbow',
#             vmin        = vmin,
#             vmax        = vmax)
#
# plt.xlabel(r'Polynomial degree')
# plt.ylabel(r'$\log( \lambda )$')
#
# plt.show()
