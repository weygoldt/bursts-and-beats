from cmath import phase

import matplotlib.pyplot as plt
import numpy as np

# simulate some data
dt = 0.001
t = np.arange(0, 1, dt)
f = 20

# generate base signal
v = np.sin(2.0 * np.pi * f * t)

# generate test signals to iterate over base signals
phase_test = np.arange(0, 2.0 * np.pi, 0.1)
v_test = [np.sin(2.0 * np.pi * f * t + p) for p in phase_test]


# make euclidean distance function
def euclidean(u, v):
    """
    euclidean computes the Euclidean distance between two arrays.

    Parameters
    ----------
    u : array
        The first array
    v : array
        The second array

    Returns
    -------
    float
        The Euclidean distance between the arrays.
    """
    return np.sqrt(np.sum(np.abs(u - v) ** 2))


# test the function
euc = euclidean(v, v_test[0])

# iterate
distances = [euclidean(v, x) for x in v_test]

# plot distances as function of phases
plt.plot(phase_test, distances)
plt.show()
