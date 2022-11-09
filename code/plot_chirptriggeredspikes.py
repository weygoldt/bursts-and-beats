import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from tqdm import tqdm

import functions as fs
from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()

# get data
d1 = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
d2 = rlx.Dataset("../data/2021-11-11-af-invivo-1.nix")

data = [d1, d2]
spike_data = []
rate_data = []
rate_stds = []
time_data = []

for d in tqdm(data):

    # extract centered spiketimes
    singlecell_spikes, time = fs.singlecell_cts(d)
    hompopulation_spikes, time = fs.hompopulation_cts(d)

    # compute gamma kde with coustom time array
    kdetime = np.linspace(
        np.min(time),
        np.max(time),
        1000,
    )

    # compute kdes for every chirp-centered spiketrain
    singlecell_kdes = [fs.causal_kde1d(x, kdetime, 0.002) for x in singlecell_spikes]

    # compute kde for population activity
    hompopulation_kdes = [
        fs.causal_kde1d(x, kdetime, 0.002) for x in hompopulation_spikes
    ]

    # compute mean kdes
    singlecell_mean = np.mean(singlecell_kdes, axis=0)
    singlecell_std = np.std(singlecell_kdes, axis=0)
    hompopulation_mean = np.mean(hompopulation_kdes, axis=0) / 6

    # sort by spike train length
    hompopulation_spikes = sorted(hompopulation_spikes, key=len, reverse=True)

    spike_data.append(hompopulation_spikes)
    rate_data.append(singlecell_mean)
    rate_stds.append(singlecell_std)
    time_data.append(kdetime)

plotheight = round((np.max(rate_data) * 1.2) / 10) * 10
fig, ax = plt.subplots(1, 2, figsize=(24 * ps.cm, 12 * ps.cm), sharex=True, sharey=True)
for a, spikes, rate, std, time in tqdm(
    zip(ax, spike_data, rate_data, rate_stds, time_data)
):

    offsets = plotheight / len(spikes)

    # remove all spikes before and after limits
    lower = -0.05
    upper = 0.1
    spikes = [np.array(s) for s in spikes]
    spikes = [np.array(s) for s in spikes]
    spikes = [s[(s > lower) & (s <= upper)] * 1000 for s in spikes]

    # cut time and rate to limits
    time = np.array(time)
    rate = rate[(time > lower) & (time <= upper)]
    # std = std[(time > lower) & (time <= upper)]
    time = time[(time > lower) & (time <= upper)] * 1000

    a.eventplot(
        spikes,
        lineoffsets=offsets,
        linelengths=offsets,
        colors="darkgrey",
        alpha=1,
    )

    a.plot([0, 0], [0, plotheight], c=ps.black, lw=1, ls="dashed")
    a.plot(time, rate, color=ps.black, lw=2)
    # a.fill_between(time, rate - std, rate + std, alpha=0.5)

    # remove upper and right axis
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)

    # make axes nicer
    a.set_xticks(np.arange(-50, 150, 25))
    a.set_yticks(np.arange(0, 55, 10))
    a.spines.left.set_bounds((0, 50))
    a.spines.bottom.set_bounds((-50, 100))

    a.margins(0.05)  # 5% padding in all directions

# add labels
fig.supxlabel("Chirp centered time [ms]", fontsize=14)
fig.supylabel("Rate [Hz]", fontsize=14)
plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.15, hspace=0)

fs.doublesave("../figures/chirp_rasterplot")
plt.show()
