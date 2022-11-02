import random
from turtle import st

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs
import plot_parameter
from termcolors import TermColor as tc

d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
ram = d["FileStimulus_3"]
stimulie_rlx = ram.stimuli
# load file stimulie from folder
ram.stimulus_folder = "../data/stimulus/"
s, t = ram.load_stimulus()

stas = []
sds = []
spike_times = []
# go through every stimulus in one repro
for stim in stimulie_rlx:
    spikes = stim.trace_data("Spikes-1")[0]
    dt = t[1] - t[0]
    time, sta, sd, count = fs.spike_triggered_average(
        spikes, s, dt, t_min=-0.025, t_max=0.025
    )
    stas.append(sta)
    spike_times.append(spikes)
    sds.append(sd)

mean_stas = np.array(np.mean(stas, axis=0))
mean_sds = np.array(np.std(sds, axis=0))

fig, ax = plt.subplots()
ax.plot(time, mean_stas)
ax.fill_between(time, mean_stas - mean_sds, mean_stas + mean_sds, alpha=0.3, zorder=-10)
ax.hlines(0, -10, 10, linestyles="dashed", alpha=0.6, color="k")
ax.vlines(0, -10, 10, linestyles="dashed", alpha=0.6, color="k")
ax.set_xlim(-0.03, 0.03)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Stimulus")
fs.doublesave("../figures/spike_triggered_average")
plt.show()
