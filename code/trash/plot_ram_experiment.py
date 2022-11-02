import random
from turtle import st

import matplotlib.pyplot as plt
import numpy as np
import plot_parameter
import rlxnix as rlx
from IPython import embed
from plotstyle import PlotStyle
from termcolors import TermColor as tc

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


ps = PlotStyle()

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
    print(len(spikes))
    dt = t[1] - t[0]
    time, snippets = spike_triggered_average(spikes, s, dt, t_min=-0.025, t_max=0.025)
    stas.extend(snippets)
    spike_times.append(spikes)

mean_stas = np.array(np.mean(stas, axis=0))
mean_sds = np.array(np.std(stas, axis=0))

fig, ax = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))

for sta in stas:
    plt.plot(time, sta, c="darkgray", alpha=0.02)
ax.plot(time, mean_stas, c=ps.black, lw=2, zorder=10)
# ax.fill_between(time, mean_stas - mean_sds, mean_stas + mean_sds, alpha=0.3, zorder=-10)

# ax.hlines(0, -10, 10, linestyles="dashed", alpha=0.6, color="k")
# ax.vlines(0, -10, 10, linestyles="dashed", alpha=0.6, color="k")

#  ax.set_xlim(-0.03, 0.03)
#  ax.set_ylim(-0.25, 0.12)

ax.set_xlabel("Stim. centered time [ms]")
ax.set_ylabel("Stimulus")

# adjust plot margings
plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.12, hspace=0)

fs.doublesave("../figures/spike_triggered_average")
plt.show()
