import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy import interpolate
from IPython import embed

import functions as fs
from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()
saveplot = True

# get data
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
fish_eodf = d.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]

reodfs = fs.sort_reodfs(d)

# get all chirp repros with multiple of 1
chirp_repros = []
for key in reodfs:
    if int(float(key)) == 1:
        chirp_repros.append(reodfs[key][0])

# find all chirp repros
# chirp_repros = [i for i in d.repros if "Chirps" in i]

# for chirp_repro in chirp_repros:

chirp_repro = chirp_repros[1]
# chirp_repro = "Chirps_5"
# chirp_no = 0
chirps = d[chirp_repro]

# collect data here
spike_t = []
centertime = []
centerchirp = []
spike_t_bursts = []
spike_t_single = []

# go through 16 trials
for i in range(chirps.stimulus_count):

    # get data
    _, time = chirps.membrane_voltage(i)
    spikes = chirps.spikes(i)
    single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(spikes, 0.01, verbose=False)
    spikes_bursts = spikes[fs.flatten(burst_spikes)]
    spike_singe = spikes[single_spikes]
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
    before_t = 0.0
    after_t = 0.05
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
        c_spikes_bursts = spikes_bursts[(spikes_bursts > tmin) & (spikes_bursts < tmax)]
        c_spikes_single = spike_singe[(spike_singe > tmin) & (spike_singe < tmax)]
        

        # get spike indices on c_time vector
        c_spike_indices = [fs.find_closest(c_time, x) for x in c_spikes]
        c_spike_indices_bursts = [fs.find_closest(c_time, x) for x in c_spikes_bursts]
        c_spike_indices_single = [fs.find_closest(c_time, x) for x in c_spikes_single]

        # make new centered time array
        c_time = time[indices] - (tmin + (tmax - tmin) / 2)

        # extract spike timestamps from centered time
        c_spikes_centered = c_time[c_spike_indices]
        c_spikes_centered_bursts = c_time[c_spike_indices_bursts]
        c_spikes_centered_single = c_time[c_spike_indices_single]

        # append centered spike times to list
        spike_t.append(c_spikes_centered)
        spike_t_bursts.append(c_spikes_centered_bursts)
        spike_t_single.append(c_spikes_centered_single)
        # spike_t.append(c_time)
        centerchirp.append(c_env)

 
embed()
exit()