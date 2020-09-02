# hw1 Excercise 1

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read data
data_set = pd.read_csv("https://raw.githubusercontent.com/mhjensen/MachineLearningMSU-FRIB2020/master/doc/pub/Regression/ipynb/datafiles/EoS.csv")

# Change column names to something more recognizable
data_set.columns = ['x', 'y']

# Fetch the x-values of the data set
x  = data_set.iloc[:, 0]

# and the y-values
y  = data_set.iloc[:, 1]

# Construct the design matrix
n = len(x)
design_matrix = np.vstack((np.ones(n), x, x**2, x**3))
design_matrix = design_matrix.T


# Linear regression

reg = LinearRegression()
reg.fit(design_matrix, y)

beta = reg.coef_


# Prediction from our model
ymodel   = np.matmul(design_matrix, beta)
df_model = pd.DataFrame(np.transpose(np.vstack((x, ymodel))), columns = ['x', 'ymodel'])


# Plot and compare

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set()

# Scatter plot of data set
sns.scatterplot(data = data_set, x = 'x', y = 'y', label = 'Data set')

# Line plot of prediction
sns.lineplot(data = df_model, x = 'x', y = 'ymodel', alpha = 0.5, label = 'Prediction')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.show()
