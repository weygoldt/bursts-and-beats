import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy import interpolate

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
chirp_repros = [reodfs[key][0] for key in reodfs if int(float(key)) == 1]
chirp_repro = chirp_repros[1]
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
    before_t = 6 * beat_p
    after_t = 6 * beat_p
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
        c_time = time[indices] - (tmin + (tmax - tmin) / 2)

        # extract spike timestamps from centered time
        c_spikes_centered = c_time[c_spike_indices]

        # append centered spike times to list
        spike_t.append(c_spikes_centered)
        # spike_t.append(c_time)
        centerchirp.append(c_env)

# compute rate
kdetime = np.linspace(c_time[0], c_time[-1], 500)
rate = np.array([fs.acausal_kde1d(s, kdetime, 0.005) for s in spike_t])
meanrate = np.mean(rate, axis=0)

# fig, ax = plt.subplots(2, 1, sharex=True)
# for chirp, spike in zip(centerchirp, spike_t):
#     ax[0].plot(c_time, chirp)
#     ax[0].scatter(spike, np.ones_like(spike) * -0.5, alpha=1)
# ax[1].plot(kdetime, meanrate)
# plt.show()

# convert spike times to ms
spike_t = [s * 1000 for s in spike_t]

if saveplot:
    height = np.max(meanrate) * 1.2
    tickscaler = np.max(meanrate) / len(spike_t)

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(24 * ps.cm, 12 * ps.cm),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    for chirp, spike in zip(centerchirp, spike_t):
        ax[0].plot(c_time * 1000, chirp, c="lightgrey", lw=1, alpha=0.1)
    ax[0].plot(c_time * 1000, centerchirp[5], c=ps.black, lw=1.5)
    ax[0].axis("off")

    # add df
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica Now Text",
        }
    )

    posx = 0.02
    posy = 0.91
    ax[0].text(
        posx,
        posy,
        r"EOD$f_{{rel}} = {}$".format(chirps.relative_eodf),
        font="stix",
        fontsize=12,
        transform=plt.gcf().transFigure,
    )
    ax[0].text(
        posx,
        posy - 0.06,
        r"$f_{{Beat}} = {}$ Hz".format(int(beat_f)),
        font="stix",
        fontsize=12,
        transform=plt.gcf().transFigure,
    )

    ax[1].eventplot(
        spike_t,
        linewidths=2,
        colors=ps.black,
    )

    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].tick_params(bottom=False)
    ax[1].set_ylabel("Trial")

    ax[1].spines.left.set_bounds((0, 100))

    ax[2].fill_between(
        kdetime * 1000,
        np.zeros_like(meanrate),
        meanrate,
        color="lightgrey",
        alpha=0.3,
        lw=0,
    )

    ax[2].plot(kdetime * 1000, meanrate, color="darkgray", lw=1)

    # remove upper and right axis
    ax[2].spines["right"].set_visible(False)
    ax[2].spines["top"].set_visible(False)

    # make axes nicer
    ax[2].set_xticks(np.arange(-200, 250, 50))
    ax[2].spines.bottom.set_bounds((-200, 200))
    ax[2].set_yticks(np.arange(0, 35, 10))
    ax[2].spines.left.set_bounds((0, 30))

    ax[2].set_ylabel("Rate [Hz]")
    ax[2].set_xlabel("Chirp centered time [ms]")

    # adjust label position
    # ax[1].xaxis.set_label_coords(0.5, -0.2)

    fig.align_ylabels(ax)

    # adjust plot margings
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.15, hspace=0.1)

    #fs.doublesave(f"../figures/chirp_triggered_spikes_{chirps.relative_eodf}")
    plt.show()
