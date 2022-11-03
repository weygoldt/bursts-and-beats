import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs
from plotstyle import PlotStyle

# import plot_parameter
from termcolors import TermColor as tc

ps = PlotStyle()

d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
ram = d["FileStimulus_3"]


def singlespike_triggered_stim(ram):

    stimuli_rlx = ram.stimuli
    ram.stimulus_folder = "../data/stimulus/"
    s, t = ram.load_stimulus()
    eodf_fisch = d.metadata["Recording"]["Subject"]["EOD Frequency"][0]

    # collect data here
    stas = []
    sds = []
    spike_times = []

    # go through every stimulus in one repro
    for stim in stimuli_rlx:

        # get spikes
        spikes = stim.trace_data("Spikes-1")[0]

        # compute bursts
        single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(
            spikes, 0.01, verbose=False
        )

        # compute burst times
        single_t = [spikes[x] for x in single_spikes]
        dt = t[1] - t[0]
        time, sta, sd, count = fs.spike_triggered_average(
            single_t, s, dt, t_min=-0.040, t_max=0.015
        )

        # collect data
        stas.append(sta)
        spike_times.append(spikes)
        sds.append(sd)

    # compute mean and std
    mean_stas = np.array(np.mean(stas, axis=0))
    mean_sds = np.array(np.std(stas, axis=0))

    # normalize
    ampl = ram[0].feature_data('gwn300Hz50s0-1_amplitude')
    stim_contrast = 0.2
    sdt_stim = stim_contrast * ampl 
    ist_std_stim = 0.3
    contrast =  ist_std_stim * (sdt_stim/ist_std_stim)
    mean_stas = np.array(np.mean(stas, axis=0))
    mean_sds = np.array(np.std(stas, axis=0))
    mean_stas = mean_stas * contrast
    mean_sds = mean_sds   * contrast


    return time, mean_stas, mean_sds


def burst_triggered_stim(ram):

    # load file stimulie from folder
    stimuli_rlx = ram.stimuli
    ram.stimulus_folder = "../data/stimulus/"
    s, t = ram.load_stimulus()
    eodf_fisch = d.metadata["Recording"]["Subject"]["EOD Frequency"][0]

    # collect data here
    stas = []
    sds = []
    spike_times = []

    # go through every stimulus in one repro
    for stim in stimuli_rlx:
        spikes = stim.trace_data("Spikes-1")[0]

        # compute bursts
        single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(
            spikes, 0.01, verbose=False
        )

        # compute burst times
        burst_t = [spikes[x[0]] for x in burst_start_stop]

        dt = t[1] - t[0]
        time, sta, sd, count = fs.spike_triggered_average(
            burst_t, s, dt, t_min=-0.040, t_max=0.015
        )
        stas.append(sta)
        spike_times.append(spikes)
        sds.append(sd)

    ampl = ram[0].feature_data('gwn300Hz50s0-1_amplitude')
    stim_contrast = 0.2
    sdt_stim = stim_contrast * ampl 
    ist_std_stim = 0.3
    contrast =  ist_std_stim * (sdt_stim/ist_std_stim)
    mean_stas = np.array(np.mean(stas, axis=0))
    mean_sds = np.array(np.std(stas, axis=0))
    mean_stas = mean_stas * contrast
    mean_sds = mean_sds   * contrast

    return time, mean_stas, mean_sds


# get data
s_time, s_mean, s_std = singlespike_triggered_stim(ram)
b_time, b_mean, b_std = burst_triggered_stim(ram)

# plot
fig, ax = plt.subplots(1, 2, figsize=(24 * ps.cm, 12 * ps.cm), sharey=True, sharex=True)

# plot burst triggered stimulus
ax[0].plot(b_time * 1000, b_mean, c=ps.black, lw=2)
ax[0].fill_between(
    b_time * 1000,
    b_mean - b_std,
    b_mean + b_std,
    color="lightgray",
    alpha=0.3,
    lw=0,
)
ax[0].plot(b_time * 1000, b_mean - b_std, c="darkgray", lw=1)
ax[0].plot(b_time * 1000, b_mean + b_std, c="darkgray", lw=1)

# plot singlespike triggered average stimulus
ax[1].plot(s_time * 1000, s_mean, c=ps.black, lw=2)
ax[1].fill_between(
    s_time * 1000,
    s_mean - s_std,
    s_mean + s_std,
    color="lightgray",
    alpha=0.3,
    lw=0,
)
ax[1].plot(s_time * 1000, s_mean - s_std, c="darkgray", lw=1)
ax[1].plot(s_time * 1000, s_mean + s_std, c="darkgray", lw=1)

ax[0].plot([-40, 15], [0, 0], ls="dashed", color=ps.black, lw=1, alpha=0.4)
ax[0].plot([0, 0], [-40, 30], ls="dashed", color=ps.black, lw=1, alpha=0.4)
ax[1].plot([-40, 15], [0, 0], ls="dashed", color=ps.black, lw=1, alpha=0.4)
ax[1].plot([0, 0], [-40, 30], ls="dashed", color=ps.black, lw=1, alpha=0.4)

for a in ax:

    # add guidelines
    a.hlines(0, -25, 25, linestyles="dashed", color="k", lw=1)
    a.vlines(0, 30, -40, linestyles="dashed", color="k", lw=1)

    # remove upper and right axis
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)

    # make axes nicer
    a.set_xticks(np.append(np.arange(0, 40, 10) - 40, np.arange(0, 15, 10)))
    a.set_yticks(np.arange(-40, 35, 10))
    a.spines.left.set_bounds((-40, 30))
    a.spines.bottom.set_bounds((-40, 15))

fig.supxlabel("Spike centered time [ms]", fontsize=14, x=0.552, y=0.009)
fig.supylabel("Average stimulus [mV/cm]", fontsize=14, x=0.02, y=0.6)

plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.14, hspace=0, wspace=0.1)
#fs.doublesave("../figures/ssta_vs_bta")

plt.show()
