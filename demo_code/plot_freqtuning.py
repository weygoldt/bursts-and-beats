import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy.signal import hilbert

import functions as fs

data = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

ram = data["FileStimulus_4"]
spikes = ram.spikes()
stim, time = ram.stimulus_output()
f = 1 / (time[1] - time[0])

# filter signal to get envelope
stim_filt = fs.bandpass_filter(np.abs(stim), f, 0.000001, 10)

# compute firing rate
spikerate = fs.causal_kde1d(spikes, time, width=0.01)

# rectify envelope
stim_filt[stim_filt <= 0.001] = 0

# find zero crossings by diff
idx = np.arange(len(stim_filt))
zero_idx = idx[stim_filt != 0]

# find gaps of continuity in index array
diffs = np.diff(zero_idx)
diffs = np.append(diffs, 0)
zerocrossings = zero_idx[diffs > 1]

# calculate boundaries
bounds = [[fs.find_closest(time, time[x] - 10), x] for x in zerocrossings]

# convert bounds to bound ranges
branges = []
for b in bounds:
    branges.append(np.arange(b[0], b[1]))

# convert spiketimes to spike indices on time array
spike_indices = [fs.find_closest(time, s) for s in spikes]

# instantaneous frequency
analytic_signal = hilbert(stim)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.append(
    np.abs(np.diff(instantaneous_phase) / (2.0 * np.pi) * f), np.nan
)

# get frequency at every spike
golist = fs.flatten(branges)
freqs = []
for s in spike_indices:
    if s in golist:
        freqs.append(instantaneous_frequency[s])

# bin both
fbins = np.linspace(0, np.max(freqs), 50)
sbins = np.linspace(0, np.max(spikerate), 50)

freqs_binned = np.digitize(freqs, fbins)
freqs_bin_means = [freqs[freqs_binned == i].mean() for i in range(1, len(fbins))]

spikerate_binned = np.digitize(spikerate, sbins)
bin_means = [spikerate[spikerate_binned == i].mean() for i in range(1, len(sbins))]
