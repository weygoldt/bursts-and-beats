import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs

# get data
d = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")
chirps = d["Chirps_24"]
v, time = chirps.membrane_voltage(0)
ctimes = chirps.chirp_times[0]

spike_t = []
before_t = 0.1
after_t = 0.2

for i in range(chirps.stimulus_count):

    # get data
    v, time = chirps.membrane_voltage(i)
    spikes = chirps.spikes(i)
    chirp_times = chirps.chirp_times[0][i]

    # compute number of indices before and after chirp to include
    dt = time[1] - time[0]
    before_indices = np.round(before_t / dt)
    after_indices = np.round(after_t / dt)

    for c in chirp_times:

        # where is chirp on time vector?
        c_index = fs.find_closest(time, c)

        # make index vector centered around chirp
        indices = np.arange(
            c_index - before_indices, c_index + after_indices + 1, dtype=int
        )

        # get max t and min t
        c_time = time[indices]
        tmin = np.min(c_time)
        tmax = np.max(c_time)

        # get spike times in this range
        c_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

        # get spike indices on c_time vector
        c_spike_indices = [fs.find_closest(c_time, x) for x in c_spikes]

        # make new centered time array
        c_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

        # extract spike timestamps from centered time
        c_spikes_centered = c_time[c_spike_indices]

        # append centered spike times to list
        spike_t.append(c_spikes_centered)


# remove empty entries
spike_t_cleaned = []
for spiketrain in spike_t:
    if len(spiketrain) == 0:
        continue
    else:
        spike_t_cleaned.append(spiketrain)


# compute gamma kde with coustom time array
kdetime = np.linspace(
    np.min(c_time),
    np.max(c_time),
    500,
)

# compute kdes
kdes = []
for spiketrain in spike_t_cleaned:
    rate = fs.causal_kde1d(spikes=spiketrain, time=kdetime, width=0.005, shape=2)
    kdes.append(rate)

# compute mean kdes
kdes = np.array(kdes)
mean_kde = []
for i in range(len(kdetime)):
    colmean = np.mean(kdes[:, i])
    mean_kde.append(colmean)

plt.eventplot(
    spike_t,
    lineoffsets=np.linspace(0, np.max(mean_kde), len(spike_t)),
    linelengths=np.ones_like(len(spike_t)) * 0.8,
    colors="black",
    alpha=1,
)
plt.axvline(0, 0, 100, color="black", linestyle="dashed")
plt.plot(kdetime, mean_kde, color="red", lw=2)
plt.show()
