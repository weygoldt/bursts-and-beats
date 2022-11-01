import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy import interpolate

import functions as fs
from termcolors import TermColor as tc

# get data
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
fish_eodf = d.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]
chirp_repro = "Chirps_1"
chirp_no = 0
chirps = d[chirp_repro]

# collect data here
spike_t = []
centertime = []
centerchirp = []


# go through 16 trials
for i in range(chirps.stimulus_count):

    # get data
    _, time = chirps.membrane_voltage(i)
    spikes = chirps.spikes(i)
    chirp_times = chirps.chirp_times[0][i]
    true_mult = chirps.relative_eodf

    # fish data
    fish_eod, fish_eodtime = chirps.eod(i)

    # signal data
    stim_eod, stim_eodtime = chirps.stimulus_output(i)
    stim_eodf = fs.rel_to_eods(fish_eodf, chirps.relative_eodf)

    # compute envelope
    beat, time = chirps.local_eod(i)

    # rectification
    lower_eod_rect = fish_eod.clip(max=0)

    # find where rectified lower EOD is now 0
    idx = np.arange(len(lower_eod_rect))
    zero_idx = idx[lower_eod_rect != 0]

    # find gaps of continuity in index array
    diffs = np.diff(zero_idx)
    diffs = np.append(diffs, 0)
    zerocrossings = zero_idx[diffs > 1]

    # calculate boundaries
    bounds = [[x, y] for x, y in zip(zerocrossings, zerocrossings[1:])]

    # calculate maxima in non-zero areas
    peaks = []
    for b in bounds:

        # make ranges from boundaries
        b_full = np.arange(b[0], b[1])
        peak = b_full[beat[b_full] == np.min(beat[b_full])][0]
        peaks.append(peak)

    # fix noisy peaks by thresholding
    peaks = np.array(peaks)[beat[peaks] < -0.2]

    # interpolate between peaks
    interp = interpolate.interp1d(
        time[peaks], beat[peaks], kind="cubic", fill_value="extrapolate"
    )
    envelope = interp(time)

    # compute beat frequency
    beat_mult = np.round(true_mult % 1, decimals=2) + 1
    stim_eodf = fish_eodf * beat_mult
    beat_f = abs(stim_eodf - fish_eodf)

    # compute beat period
    beat_p = 1 / beat_f

    # compute number of indices before and after chirp based on beat period
    before_t = 2 * beat_p
    after_t = 2 * beat_p
    dt = time[1] - time[0]
    before_indices = np.round(before_t / dt)
    after_indices = np.round(after_t / dt)

    # get all chirps
    for c in chirp_times:

        # where is chirp on time vector?
        c_index = fs.find_closest(time, c)

        # make index vector centered around chirp
        indices = np.arange(
            c_index - before_indices, c_index + after_indices, dtype=int
        )
        c_time = time[indices]
        c_env = envelope[indices]

        # get tmin and tmax
        tmin = np.min(c_time)
        tmax = np.max(c_time)

        # get spike times in this range
        c_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

        # get spike indices on c_time vector
        c_spike_indices = [fs.find_closest(c_time, x) for x in c_spikes]

        # make new centered time array
        c_time = c_time = time[indices] - tmin

        # extract spike timestamps from centered time
        c_spikes_centered = c_time[c_spike_indices]

        # append centered spike times to list
        spike_t.append(c_spikes_centered)
        # spike_t.append(c_time)
        centerchirp.append(c_env)

for chirp, spike in zip(centerchirp, spike_t):
    plt.plot(c_time, chirp)
    plt.scatter(spike, np.ones_like(spike) * -0.5, alpha=1)
plt.show()
