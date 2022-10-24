import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs

# load data
d = rlx.Dataset("../data/2022-10-20-ab-invivo-1.nix")

# extract baseline
bl = d["BaselineActivity_2"]

# extract voltage and time
spike_times = bl.spikes()
v, t = bl.membrane_voltage()

# compute interspike intervals
isi = fs.isis([spike_times])[0]

# set isi threshold
thresh = 0.020

# find indices of spike times in time array
spike_time_indices = [fs.find_closest(t, x) for x in spike_times]
isi_indices = np.arange(len(isi) + 1)

# find spikes where at least one sourrounding isi is lower than the threshold
burst_spikes = []
single_spikes = []
burst = False

for isi_index in isi_indices:

    # first spike
    if isi_index == isi_indices[0]:
        spike_isi = isi[0]

        # test if greater than thresh
        if spike_isi < thresh:
            burst = True
        else:
            burst = False

    # last spike
    elif isi_index == isi_indices[-1]:
        spike_isi = isi[-1]

        # test if greater than thresh
        if spike_isi < thresh:
            burst = True
        else:
            burst = False

    # middle spikes
    else:
        spike_isi = isi[isi_index - 1 : isi_index + 1]

        # test if greater than thresh
        if (spike_isi[0] < thresh) or (spike_isi[1] < thresh):
            burst = True
        else:
            burst = False

    if burst:
        burst_spikes.append(isi_index)
    else:
        single_spikes.append(isi_index)


# find indices of spike times (bursting and non bursting) in time array
burst_time_indices = [fs.find_closest(t, x) for x in spike_times[burst_spikes]]
single_time_indices = [fs.find_closest(t, x) for x in spike_times[single_spikes]]

diffs = np.diff(burst_spikes)
gaps = np.where(diffs != 1)[0]

# plot
plt.plot(t, v)
plt.scatter(
    t[single_time_indices],
    np.ones_like(single_time_indices) - 21,
    color="green",
    marker="|",
)
plt.scatter(
    t[burst_time_indices],
    np.ones_like(burst_time_indices) - 20,
    color="red",
    marker="|",
)
plt.scatter(
    spike_times[np.array(burst_spikes)[gaps]],
    np.ones_like(spike_times[np.array(burst_spikes)[gaps]]) - 19,
    color="cornflowerblue",
)
plt.show()
