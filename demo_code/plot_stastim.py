import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs


def spike_triggered_average(spikes, stimulus, dt, t_min=-0.01, t_max=0.01):

    time = np.arange(t_min, t_max, dt)  # time for the STA
    snippets = []

    count = 0
    for t in spikes:
        min_index = int(np.round((t + t_min) / dt))
        max_index = min_index + len(time)

        if (min_index < 0) or (max_index > len(stimulus)):
            continue

        snippets.append(stimulus[min_index:max_index])  # store snippet
        count += 1

    return time, snippets


data = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

ram = data["FileStimulus_4"]
spikes = ram.spikes()
stim, time = ram.stimulus_output()
f = 1 / (time[1] - time[0])

# filter signal to get envelope
stim_filt = fs.bandpass_filter(np.abs(stim), f, 0.000001, 10)

# rectify envelope
stim_filt[stim_filt <= 0.001] = 0

# find zero crossings by diff
idx = np.arange(len(stim_filt))
nonzero_idx = idx[stim_filt != 0]

# find gaps of continuity in index array
diffs = np.diff(nonzero_idx)
diffs = np.append(diffs, 0)
zerocrossings = nonzero_idx[diffs > 1]

# calculate boundaries
bounds = [[fs.find_closest(time, time[x] - 10), x] for x in zerocrossings]

# convert bounds to bound ranges
branges = [np.arange(b[0], b[1]) for b in bounds]

# convert spiketimes to spike indices on time array
spike_indices = [fs.find_closest(time, s) for s in spikes]

dt = time[1] - time[0]
stas = []
for b in branges:

    # get all valid spikes, i.e. spikes during stimulus
    valid_spike_indices = list(set(b).intersection(spike_indices))

    # compute sta for trial
    newt, snippets = spike_triggered_average(time[valid_spike_indices], stim, dt)

    # collect
    stas.extend(snippets)

stas = np.array(stas)
meansta = np.mean(stas, axis=0)
stdsta = np.std(stas, axis=0)

plt.plot(newt, meansta)
plt.fill_between(newt, meansta - stdsta, meansta + stdsta, alpha=0.5)
plt.axvline(0, 0, 1, lw=1, ls="dashed", c="k", alpha=0.3)
plt.axhline(0, 0, 1, lw=1, ls="dashed", c="k", alpha=0.3)
plt.show()
