import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()

# get data
d1 = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
d2 = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")

data = [d1, d2]

fig, ax = plt.subplots(1, 2, figsize=(24 * ps.cm, 12 * ps.cm))

for a, d in zip(ax, data):

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
    hompopulation_mean = np.mean(hompopulation_kdes, axis=0) / 6

    # sort by spike train length
    hompopulation_spikes = sorted(hompopulation_spikes, key=len, reverse=True)

    # plot
    a.eventplot(
        hompopulation_spikes,
        lineoffsets=0.05,
        linelengths=0.05,
        colors="black",
        alpha=1,
    )

    a.axvline(0, 0, 100, color="black", linestyle="dashed", lw=1)
    a.plot(kdetime, hompopulation_mean, color="green", lw=2)
    a.plot(kdetime, singlecell_mean, color="red", lw=2)
    # ax.set_xlim(-0.08, 0.18)

plt.show()
