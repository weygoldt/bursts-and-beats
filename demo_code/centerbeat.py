import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs
from termcolors import TermColor as tc

################################################################################
# TO DO
# Find out why i = 5 is skipped for all repros (something with the envelope
# computation does not work)
################################################################################

# get data
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

# find all chirp repros
chirp_repros = [i for i in d.repros if "Chirps" in i]
# chirp_repros = chirp_repros[]

# collect beat-centered spike times here
spike_t = []
times = []

# go through all chirp repros
for repro in chirp_repros:

    # get chirps from each repro
    chirps = d[repro]
    chirp_duration = chirps.metadata["RePro-Info"]["settings"]["chirpwidth"][0][0]
    fish_eodf = d.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]

    for i in range(chirps.stimulus_count):

        # get data
        _, time = chirps.membrane_voltage(i)

        if np.max(time) < 1:
            print(tc.err("Trial too short, skipping"))
            continue

        dt = time[1] - time[0]
        rate = 1 / dt
        spikes = chirps.spikes(i)
        chirp_times = chirps.chirp_times[0][i]
        nchirps = len(chirp_times)
        stim, stimtime = chirps.stimulus_output(i)

        # fish data
        fish_eod, fish_eodtime = chirps.eod(i)

        # signal data
        stim_eod, stim_eodtime = chirps.stimulus_output(i)
        stim_eodf = fs.rel_to_eods(fish_eodf, chirps.relative_eodf)

        # make envelope
        beat, envelope, envelope_time = fs.beat_envelope(
            stim_eod, fish_eod, stim_eodf, fish_eodf, time
        )

        # Find chirp areas -------------------------------------------------------------

        # find index of chirp
        chirp_indices = [fs.find_closest(time, c) for c in chirp_times]

        # make window around chirp according to chirp duration
        chirp_windows = [
            [c - chirp_duration / 2, c + chirp_duration / 2] for c in chirp_times
        ]

        # find indices of chirp window start and stop
        chirp_window_indices = [
            [fs.find_closest(time, x[0]), fs.find_closest(time, x[1])]
            for x in chirp_windows
        ]

        # convert indices to NON-chirp window start stop
        nonchirp_indices = []
        for i, w in enumerate(chirp_window_indices):

            # start first window with 0
            if w[0] == chirp_window_indices[0][0]:
                nonchirp_indices.append([0, w[0]])

            # end last window with len(time)
            elif w[1] == chirp_window_indices[-1][1]:
                nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])
                nonchirp_indices.append([w[1], len(time)])

            # take last of previous and first of current for middle windows
            else:
                nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])

        # split envelope in non chirp periods
        env_split = [envelope[x[0] : x[1]] for x in nonchirp_indices]
        env_time = [envelope_time[x[0] : x[1]] for x in nonchirp_indices]

        # compute peak timestamps by sine approximation for each envelope snippet
        try:
            env_peaks = fs.flatten(
                [
                    t[fs.envelope_peaks(env, t, rate)]
                    for env, t in zip(env_split, env_time)
                ]
            )
        except:
            embed()

        # convert peak timestamps to indices on whole time array
        beat_peaks = [fs.find_closest(time, x) for x in env_peaks]

        # draw random beat peaks
        selected_beats = random.sample(beat_peaks, nchirps)

        # Center the time at the beat peak ---------------------------------------------

        # compute number of indices before and after chirp to include
        before_t = 0.05
        after_t = 0.1
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
