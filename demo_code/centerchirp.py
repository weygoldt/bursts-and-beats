import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from termcolors import TermColor as tc


def singlecell_cts(data):
    """
    singlecell_cts extracts the spikes centered around the chirp stimulus
    (chirp-triggered spikes) onset for every single chirp stimulus.

    Parameters
    ----------
    data : rlxnix dataset
        Relacs dataset imported with rlxnix

    Returns
    -------
    spike_t : array of arrays
        an array including an array for every ct-spiketrain
    c_time : array
        the time centered around the chirp
    """

    # collect chirp-centered spike times here
    spike_t = []

    # padding around chirp
    before_t = 0.1
    after_t = 0.2

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = d[repro]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)
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
                    c_index - before_indices, c_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    c_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    continue

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

    return spike_t, c_time


def hompopulation_cts(data):
    # collect chirp-centered spike times here
    spike_t = []

    # padding around chirp
    before_t = 0.1
    after_t = 0.2

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = d[repro]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)
            spikes = chirps.spikes(i)
            chirp_times = chirps.chirp_times[0][i]

            # compute number of indices before and after chirp to include
            dt = time[1] - time[0]
            before_indices = np.round(before_t / dt)
            after_indices = np.round(after_t / dt)

            # collect spikes for this trial here
            spikelist = []

            for c in chirp_times:

                # where is chirp on time vector?
                c_index = fs.find_closest(time, c)

                # make index vector centered around chirp
                indices = np.arange(
                    c_index - before_indices, c_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    c_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    continue

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
                spikelist.append(c_spikes_centered)

            # flatten spike list to simulate activity of hom population
            spikelist_flat = fs.flatten(spikelist)

            # save to spike times list
            spike_t.append(spikelist_flat)

    return spike_t, c_time


# get data
d = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")

# extract centered spiketimes
singlecell_spikes, time = singlecell_cts(d)
hompopulation_spikes, time = hompopulation_cts(d)

# compute gamma kde with coustom time array
kdetime = np.linspace(
    np.min(time),
    np.max(time),
    1000,
)

# compute kdes for every chirp-centered spiketrain
singlecell_kdes = [fs.causal_kde1d(x, kdetime, 0.002) for x in singlecell_spikes]

# compute kde for population activity
hompopulation_kdes = [fs.causal_kde1d(x, kdetime, 0.002) for x in hompopulation_spikes]

# compute mean kdes
# singlecell_kdes = np.array(singlecell_kdes)
singlecell_mean = np.mean(singlecell_kdes, axis=0)
hompopulation_mean = np.mean(hompopulation_kdes, axis=0)
# chirpspikes_median = np.median(singlecell_kdes, axis=0)
# chirpspikes_q5s = np.quantile(singlecell_kdes, 0.05, axis=0)
# chirpspikes_q95s = np.quantile(singlecell_kdes, 0.95, axis=0)

fig, ax = plt.subplots()
ax.eventplot(
    hompopulation_spikes,
    lineoffsets=0.3,
    linelengths=1,
    colors="darkgrey",
    alpha=1,
)

ax.axvline(0, 0, 100, color="black", linestyle="dashed", lw=1)
ax.plot(kdetime, hompopulation_mean, color="green", lw=2)
plt.show()
