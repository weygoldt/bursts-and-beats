import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from termcolors import TermColor as tc

# get data
d = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")

# find all chirp repros
chirp_repros = [i for i in d.repros if "Chirps" in i]
# chirp_repros = chirp_repros[]

# collect beat-centered spike times here
spike_t = []

# go through all chirp repros
for repro in chirp_repros:

    # get chirps from each repro
    chirps = d[repro]
    chirp_duration = chirps.metadata["RePro-Info"]["settings"]["chirpwidth"][0][0]
    fish_eodf = d.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]

    for i in range(chirps.stimulus_count):

        # get data
        _, time = chirps.membrane_voltage(i)
        spikes = chirps.spikes(i)
        chirp_times = chirps.chirp_times[0][i]
        nchirps = len(chirp_times)
        stim, stimtime = chirps.stimulus_output(i)

        # fish data
        fish_eod, fish_eodtime = chirps.eod(i)

        # signal data
        stim_eod, stim_eodtime = chirps.stimulus_output(i)
        stim_eodf = fs.rel_to_eods(fish_eodf, chirps.relative_eodf)

        # Find beat peaks --------------------------------------------------------------

        # make envelope
        beat, envelope, envelope_time = fs.beat_envelope(
            stim_eod, fish_eod, stim_eodf, fish_eodf, time
        )

        # get envelope sampling rate (should be the same as data)
        rate = 1 / (envelope_time[1] - envelope_time[0])

        # bandpass filter envelope
        env_filt = fs.bandpass_filter(envelope, rate, 10, 50)

        # rectify envelope
        env_filt[env_filt < 0] = 0

        plt.plot(envelope_time, env_filt)
        plt.show()

        # find where rectified lower EOD is now 0
        idx = np.arange(len(env_filt))
        zero_idx = idx[env_filt != 0]

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
            peak = b_full[env_filt[b_full] == np.max(env_filt[b_full])][0]
            peaks.append(peak)

        # Find chirp areas -------------------------------------------------------------

        # find index of chirp
        chirp_indices = [fs.find_closest(time, c) for c in chirp_times]

        # make window around chirp according to chirp duration
        chirp_windows = [
            [c - chirp_duration / 2, c + chirp_duration / 2] for c in chirp_times
        ]

        # find indices of chirp windows
        chirp_window_indices = fs.flatten(
            [
                np.arange(fs.find_closest(time, x[0]), fs.find_closest(time, x[1]))
                for x in chirp_windows
            ]
        )

        # compute the time windows where chirps are NOT
        indices = np.arange(len(time))
        indices = np.delete(indices, chirp_window_indices)

        # Draw random beat peaks in non-chirp areas ------------------------------------

        # select all beat peaks that are not in chirp range
        verified_peaks = [p for p in peaks if p in indices]

        # draw random beat peaks
        selected_beats = random.sample(verified_peaks, nchirps)

        # Center the time at the beat peak ---------------------------------------------

        # compute number of indices before and after chirp to include
        before_t = 0.1
        after_t = 0.2
        dt = time[1] - time[0]
        before_indices = np.round(before_t / dt)
        after_indices = np.round(after_t / dt)

        for sb in time[selected_beats]:

            # where is index on the time vector?
            b_index = fs.find_closest(time, sb)

            # make index vector centered around beat
            indices = np.arange(
                b_index - before_indices, b_index + after_indices, dtype=int
            )

            # get max t and min t
            try:
                b_time = time[indices]
            except:
                print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                print(f"max index: {np.max(indices)}")
                print(f"time length: {len(time)}")
                continue

            tmin = np.min(b_time)
            tmax = np.max(b_time)

            # get spike times in this range
            b_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

            # get spike indices on b_time vector
            b_spike_indices = [fs.find_closest(b_time, x) for x in b_spikes]

            # make new centered time array
            b_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

            # extract spike timestamps from centered time
            b_spikes_centered = b_time[b_spike_indices]

            # append centered spike times to list
            spike_t.append(b_spikes_centered)


plt.eventplot(spike_t)
plt.show()
