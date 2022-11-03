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
count_a = 0
count_s = 0
count_b = 0

# go through every stimulus, compute sta, resonstruct
for stim in stimuli_rlx:

    # get data
    spikes = stim.trace_data("Spikes-1")[0]
    dt = t[1] - t[0]
    duration = t[-1] - t[0]

    # extract singles and bursts
    single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(
        spikes, 0.01, verbose=True
    )
    burst_t = [spikes[x[0]] for x in burst_start_stop]
    burst_ta = fs.flatten([spikes[x] for x in burst_start_stop])
    sspike_t = [spikes[x] for x in single_spikes]
    count_a += len(spikes)
    count_s += len(single_spikes)
    count_b += len(burst_start_stop)

    # compute sta
    time, sta_a, _, _ = fs.spike_triggered_average(
        spikes, s, dt, t_min=-0.025, t_max=0.025
    )
    time, sta_s, _, _ = fs.spike_triggered_average(
        sspike_t, s, dt, t_min=-0.025, t_max=0.025
    )
    time, sta_b, _, _ = fs.spike_triggered_average(
        burst_t, s, dt, t_min=-0.025, t_max=0.025
    )

    # reconstruct
    est_a = fs.reconstruct_stimulus(spikes, sta_a, duration, dt)
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

# plot euclidean distance boxplots
height1 = 0  # x where points go
jitter_width = 0.02  # width of jittered data

jit_x1 = np.random.normal(height1, jitter_width, size=len(euc_a))
jit_x2 = np.random.normal(height1, jitter_width, size=len(euc_s))
jit_x3 = np.random.normal(height1, jitter_width, size=len(euc_b))

fig, ax = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))
bp = ax.boxplot([euc_a, euc_s, euc_b], positions=[1.1, 1.6, 2.1], showfliers=False)
for median in bp["medians"]:
    median.set_color("black")
ax.scatter(jit_x1 + 0.9, euc_a, marker=".", color=ps.black)
ax.scatter(jit_x2 + 1.4, euc_s, marker=".", color=ps.black)
ax.scatter(jit_x3 + 1.9, euc_b, marker=".", color=ps.black)

ticks = [1, 1.5, 2]
labels = ["All spikes", "Single spikes", "Burst"]
ax.set_xticks(ticks=ticks, labels=labels)
ax.set_ylabel("Euclidean distance")

# make axes nicer
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks(np.arange(1, 2.5, 0.5))
ax.set_yticks(np.arange(188, 201, 2))
ax.spines.left.set_bounds((188, 200))
ax.spines.bottom.set_bounds((0.8, 2.2))

plt.subplots_adjust(left=0.12, right=1.1, top=0.99, bottom=0.08, hspace=0.2)
fs.doublesave("../figures/burst_eucdist")
plt.show()


# compute means & normalize
mean_a = np.mean(estis_all, axis=0)
mean_s = np.mean(estis_s, axis=0)
mean_b = np.mean(estis_b, axis=0)

# scale up
ds = np.max(s) - np.min(s)
dr_a = np.max(mean_a) - np.min(mean_a)
dr_s = np.max(mean_s) - np.min(mean_s)
dr_b = np.max(mean_b) - np.min(mean_b)
frac_a = ds / dr_a
frac_s = ds / dr_s
frac_b = ds / dr_b
mean_a = mean_a * frac_a
mean_s = mean_s * frac_s
mean_b = mean_b * frac_b

# plot reconstruct vs stim

# get nice window
start = fs.find_closest(t, 4.45)
stop = fs.find_closest(t, 4.65)

# scale time to ms
t = t[:-1][start:stop] * 1000
t = t - np.min(t)

fig, ax = plt.subplots(3, 1, figsize=(24 * ps.cm, 12 * ps.cm), sharex=True, sharey=True)

l1 = ax[0].plot(t, s[:-1][start:stop], c=ps.black, alpha=0.3, label="Stimulus")
l2 = ax[0].plot(t, mean_a[start:stop], c=ps.gblue1, lw=2, label="All spikes")
l3 = ax[1].plot(t, s[:-1][start:stop], c=ps.black, alpha=0.3, label="_nolegend_")
l4 = ax[1].plot(t, mean_s[start:stop], c=ps.gblue2, lw=2, label="Single spikes")
l5 = ax[2].plot(t, s[:-1][start:stop], c=ps.black, alpha=0.3, label="_nolegend_")
l6 = ax[2].plot(t, mean_b[start:stop], c=ps.gblue3, lw=2, label="Bursts")

for a in ax[:-1]:

    # remove upper and right axis
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.tick_params(bottom=False)

# make axes nicer
ax[2].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[2].set_xticks(np.arange(0, 205, 25))
ax[2].set_yticks(np.arange(-1, 1.1, 1))
ax[2].spines.left.set_bounds((-1, 1))
ax[2].spines.bottom.set_bounds((0, 200))

fig.legend(
    [l1, l2, l3, l6],
    labels=["Stimulus", "All spikes", "Single spikes", "Bursts"],
    ncol=4,
    loc="upper center",
)

plt.subplots_adjust(left=0.08, right=1, top=0.9, bottom=0.15, hspace=0.2)

fig.supxlabel("Time [ms]", fontsize=14)
fig.supylabel("Amplitude [mV/cm]", fontsize=14)
fs.doublesave("../figures/stim_reconstr")
plt.show()
