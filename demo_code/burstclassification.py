import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from termcolors import TermColor as tc

# load data
d = rlx.Dataset("data/2022-10-20-ab-invivo-1.nix")

# extract baseline
bl = d["BaselineActivity_2"]

# extract voltage and time
spike_times = bl.spikes()
v, t = bl.membrane_voltage()

# set isi threshold
thresh = 0.016

# detect bursts
single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(spike_times, thresh)

# plotting
burst_spikes_flat = fs.flatten(burst_spikes)
plt.plot(t, v)
plt.scatter(
    spike_times[single_spikes],
    np.ones_like(spike_times[single_spikes]) - 21,
    marker="|",
)
plt.scatter(
    spike_times[burst_spikes_flat],
    np.ones_like(spike_times[burst_spikes_flat]) - 20,
    marker="|",
)
for bounds in burst_start_stop:
    plt.axvspan(spike_times[bounds[0]], spike_times[bounds[1]], alpha=0.1, color="red")
plt.show()
