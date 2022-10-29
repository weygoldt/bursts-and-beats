import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy.stats import gamma

import functions as fs


def causal_kde1d(spikes, time, width, shape=2):
    """
    causalkde computes a kernel density estimate using a causal kernel (i.e. exponential or gamma distribution).
    A shape of 1 turns the gamma distribution into an exponential.

    Parameters
    ----------
    spikes : array-like
        spike times
    time : array-like
        sampling time
    width : float
        kernel width
    shape : int, optional
        shape of gamma distribution, by default 1

    Returns
    -------
    rate : array-like
        instantaneous firing rate
    """

    # compute dt
    dt = time[1] - time[0]

    # time on which to compute kernel:
    tmax = 10 * width

    # kernel not wider than time
    if 2 * tmax > time[-1] - time[0]:
        tmax = 0.5 * (time[-1] - time[0])

    # kernel time
    ktime = np.arange(-tmax, tmax, dt)

    # gamma kernel centered in ktime:
    kernel = gamma.pdf(
        x=ktime,
        a=shape,
        loc=0,
        scale=width,
    )

    # indices of spikes in time array:
    indices = np.asarray((spikes - time[0]) / dt, dtype=int)

    # binary spike train:
    brate = np.zeros(len(time))
    brate[indices[(indices >= 0) & (indices < len(time))]] = 1.0

    # convolution with kernel:
    rate = np.convolve(brate, kernel, mode="same")

    return rate


# get data
d = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")
chirps = d["Chirps_24"]
v, time = chirps.membrane_voltage()
spikes = chirps.spikes()

width = 0.005
rate = causal_kde1d(spikes=spikes, time=time, width=width, shape=2)

# plot
plt.plot(time, rate)
plt.plot(time, v)
plt.show()
