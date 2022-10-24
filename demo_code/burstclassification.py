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
thresh = 0.02

# find indices of spike times in time array
spike_time_indices = [fs.find_closest(t, x) for x in spike_times]
spike_spike_indices = np.arange(len(spike_times))
isi_indices = np.arange(len(isi))

# find spikes where at least one sourrounding isi is lower than the threshold
burst_spikes = []
single_spikes = []
burst = False
switch = False

for spike in spike_spike_indices:

    # first spike
    if spike == spike_spike_indices[0]:
        spike_isi = isi[0]

        # test if greater than thresh
        if spike_isi < thresh:
            if burst == True:
                burst_list.append(spike)
            else:
                burst_list = []
                burst_list.append(spike)
                burst = True
        elif burst and spike_isi >= thresh:
            burst = False
            switch = True
        else:
            burst = False

    # last spike
    elif spike == spike_spike_indices[-1]:
        spike_isi = isi[-1]

        # test if greater than thresh
        if spike_isi < thresh:
            if burst == True:
                burst_list.append(spike)
            else:
                burst_list = []
                burst_list.append(spike)
                burst = True
        elif burst == True and spike_isi >= thresh:
            burst = False
            switch = True
        else:
            burst = False

    # middle spikes
    else:
        spike_isi = isi[spike - 1 : spike + 1]

        # test if greater than thresh
        if (spike_isi[0] < thresh) or (spike_isi[1] < thresh):
            if burst == True:
                burst_list.append(spike)
            else:
                burst_list = []
                burst_list.append(spike)
                burst = True
        elif burst and ((spike_isi[0] >= thresh) or (spike_isi[1] >= thresh)):
            burst = False
            switch = True
        else:
            burst = False

    if switch:
        burst_spikes.append(burst_list)

    if burst == False:
        single_spikes.append(spike)

    switch = False

# convert to numpy arrays
burst_spikes = np.array(burst_spikes, dtype=object)
burst_spikes_flat = np.array(fs.flatten(burst_spikes))

burst_start_stop = []
for burst in burst_spikes:
    burst_start_stop.append([burst[0], burst[-1]])
burst_start_stop = np.array(burst_start_stop)

single_spikes = np.array(single_spikes)

# find indices of spike times (bursting and non bursting) in time array
# burst_time_indices = [fs.find_closest(t, x) for x in spike_times[burst_spikes]]
# single_time_indices = [fs.find_closest(t, x) for x in spike_times[single_spikes]]

# gaps = np.where(abs(np.diff(burst_spikes)) > 1)[0] + 1

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
    plt.axvspan(spike_times[bounds[0]], spike_times[bounds[1]], alpha=0.5, color="red")
plt.show()
