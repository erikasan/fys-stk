import numpy as np

def bootstrap(data, statistic, B, uncertainty = False):
    """
    Estimates some statistic of an array of data using the bootstrap method
    Also gives the uncertainty of said statistic

    Args:
        data:      numpy.ndarray, the data set from which to calculate some statistic
        statistic: function
        B:         Number of bootstrap samples

    """

    n = np.size(data)

    statistics = np.zeros(B)

    for i in range(B):
        sample = np.zeros(n)

        for j in range(n):
            random_index = np.random.randint(0, n)
            sample[j]    = data[random_index]

        statistics[i] = statistic(sample)

    if uncertainty:
        unbiased_var = B/(B - 1)*np.var(statistics)
        unbiased_std = np.sqrt(unbiased_var)

        uncertainty = 2.576*unbiased_std # 99% confidence interval

        return np.mean(statistics), uncertainty

    else:
        return np.mean(statistics)

# Examples of use + validation

"""
# Generate an array of 1's
In [2]: data = np.ones(100)

# The mean should obviously be 1
In [3]: bootstrap(data, np.mean, 100)
Out[3]: (1.0, 0.0)

# And the standard deviation should obviously be 0
In [4]: bootstrap(data, np.std, 100)
Out[4]: (0.0, 0.0)

"""

"""
# Draw 1000 samples from the standard normal distribution (mean = 0, std = 1)
In [5]: data = np.random.randn(1000)

In [6]: bootstrap(data, np.std, 100)
Out[6]: (0.9955374262402512, 0.057856296976316925)

In [7]: data = np.random.randn(1000)

In [8]: bootstrap(data, np.mean, 100)
Out[8]: (0.00949586913589916, 0.0901332777735257)

In [9]: bootstrap(data, np.std, 100)
Out[9]: (1.0040066770603195, 0.06045230089576016)

# Compare with just taking the sample mean and sample std directly
In [10]: np.mean(data)
Out[10]: 0.004276396707743389

In [11]: np.std(data)
Out[11]: 1.0099285899017396

"""
