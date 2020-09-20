import numpy as np

def bootstrap(data, statistic, B):
    """
    Estimates some statistic of an array of data using the bootstrap method

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

    return np.mean(statistics)


# Examples of use + validation

"""
# Generate an array of 1's
In [25]: data = np.ones(100)

# The mean should obviously be 1
In [26]: bootstrap(data, np.mean, 100)
Out[26]: 1.0

# And the standard deviation should obviously be 0
In [27]: bootstrap(data, np.std, 100)
Out[27]: 0.0
"""

"""
# Draw 1000 samples from the standard normal distribution (mean = 0, std = 1)
In [48]: data = np.random.randn(1000)

In [49]: bootstrap(data, np.mean, 100)
Out[49]: 0.018391173995286916

In [50]: bootstrap(data, np.std, 100)
Out[50]: 0.9795808160248969

# Compare with just taking the mean and std directly
In [51]: np.mean(data)
Out[51]: 0.021858823040110907

In [52]: np.std(data)
Out[52]: 0.9822506888335921
"""
