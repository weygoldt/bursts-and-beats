import time

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs

# load data
d = rlx.Dataset("../data/2022-10-20-ab-invivo-1.nix")

# extract baseline
bl = d["BaselineActivity_2"]

# extract voltage and time
spike_times = bl.spikes()
v, t = bl.membrane_voltage()

# filter signal to cancel slow changes in membrane potential
rate = 1 / (t[1] - t[0])  # in Hz
nqst = rate / 2  # Niquist frequency
v_filt = fs.bandpass_filter(v, rate, 0.2, nqst * 0.99)

# dynamic threshold

# spike_indices = set(t).intersection(spike_times)
spike_indices = [list(t).index(x) for x in spike_times]

spike_v = v_filt

plt.plot(t, v_filt)
plt.show()
