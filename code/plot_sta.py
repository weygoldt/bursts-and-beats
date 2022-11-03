import random
from turtle import st

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
stimulie_rlx = ram.stimuli
# load file stimulie from folder
ram.stimulus_folder = "../data/stimulus/"
s, t = ram.load_stimulus()
eodf_fisch = d.metadata["Recording"]["Subject"]["EOD Frequency"][0]

stas = []
sds = []
spike_times = []
# go through every stimulus in one repro
for stim in stimulie_rlx:
    spikes = stim.trace_data("Spikes-1")[0]
    dt = t[1] - t[0]
    time, sta, sd, count = fs.spike_triggered_average(
        spikes, s, dt, t_min=-0.04, t_max=0.015
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

fig, ax = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))
ax.plot(time * 1000, mean_stas, color=ps.black, lw=2)
ax.fill_between(
    time * 1000,
    mean_stas - mean_sds,
    mean_stas + mean_sds,
    alpha=0.3,
    zorder=-10,
    color="lightgray",
    lw=0,
)
ax.plot(time * 1000, mean_stas - mean_sds, color="darkgray", lw=1)
ax.plot(time * 1000, mean_stas + mean_sds, color="darkgray", lw=1)

ax.plot([-40, 15], [0, 0], ls="dashed", color=ps.black, lw=1, alpha=0.4)
ax.plot([0, 0], [-25, 10], ls="dashed", color=ps.black, lw=1, alpha=0.4)
ax.set_xlim(-42, 17)

ax.set_xlabel("Time [ms]")
ax.set_ylabel("Average stimulus [mV/cm]")

# remove upper and right axis
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# make axes nicer
ax.set_xticks(np.arange(-40, 20, 5))
ax.set_yticks(np.arange(-25, 15, 5))
ax.spines.left.set_bounds((-25, 10))
ax.spines.bottom.set_bounds((-40, 15))

plt.subplots_adjust(left=0.12, right=0.98, top=0.99, bottom=0.12, hspace=0)

fs.doublesave("../figures/spike_triggered_average")
plt.show()
