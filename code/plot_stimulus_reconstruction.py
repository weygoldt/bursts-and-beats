import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs
from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()

# get data
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
ram = d["FileStimulus_3"]
stimuli_rlx = ram.stimuli

# load file stimulie from folder
ram.stimulus_folder = "../data/stimulus/"
s, t = ram.load_stimulus()
eodf_fisch = d.metadata["Recording"]["Subject"]["EOD Frequency"][0]

# collect data here
estis_all = []
estis_s = []
estis_b = []
stims = []

# go through every stimulus, compute sta, resonstruct
for stim in stimuli_rlx:

    # get data
    spikes = stim.trace_data("Spikes-1")[0]
    dt = t[1] - t[0]
    duration = t[-1] - t[0]

    # extract singles and bursts
    single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(
        spikes, 0.01, verbose=False
    )
    burst_t = [spikes[x[0]] for x in burst_start_stop]
    sspike_t = [spikes[x] for x in single_spikes]

    # compute sta
    time, sta_all, _, _ = fs.spike_triggered_average(
        spikes, s, dt, t_min=-0.025, t_max=0.025
    )
    time, sta_s, _, _ = fs.spike_triggered_average(
        sspike_t, s, dt, t_min=-0.025, t_max=0.025
    )
    time, sta_b, _, _ = fs.spike_triggered_average(
        burst_t, s, dt, t_min=-0.025, t_max=0.025
    )

    # reconstruct
    est_a = fs.reconstruct_stimulus(spikes, sta_all, duration, dt)
    est_s = fs.reconstruct_stimulus(spikes, sta_s, duration, dt)
    est_b = fs.reconstruct_stimulus(spikes, sta_b, duration, dt)

    # collect data
    estis_all.append(est_a)
    estis_s.append(est_s)
    estis_b.append(est_b)
    stims.append(s)

# compute euclidean distances
euc_a = [fs.euclidean(s[:-1], x) for x in estis_all]
euc_s = [fs.euclidean(s[:-1], x) for x in estis_s]
euc_b = [fs.euclidean(s[:-1], x) for x in estis_b]

plt.boxplot(euc_a, positions=[1])
plt.boxplot(euc_s, positions=[1.5])
plt.boxplot(euc_b, positions=[2])
ticks = [1, 1.5, 2]
labels = ["all", "single", "burst"]
plt.xticks(ticks=ticks, labels=labels)
plt.ylabel("euclidean distance")
plt.show()
