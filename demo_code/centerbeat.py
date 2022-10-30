import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from termcolors import TermColor as tc

################################################################################
# TO DO:
# - Find out why i = 5 is skipped for all repros (something with the envelope
# computation does not work).
# - Skip the envelope problem for now and find out why we dont see a response.
# - Looked at some spike trains. I think there is no response because the cell did not respond.
################################################################################

# get data
data = rlx.Dataset("../data/data_2021/2021-11-11-af-invivo-1.nix")

# extract centered beats
singlecell_spikes, time = fs.singlecell_bts(data)
hompopulation_spikes, time = fs.hompopulation_bts(data)

# compute gamma kde with coustom time array
kdetime = np.linspace(
    np.min(time),
    np.max(time),
    1000,
)

# compute kdes for every chirp-centered spiketrain
singlecell_kdes = [fs.causal_kde1d(x, kdetime, 0.01) for x in singlecell_spikes]

# compute kde for population activity
hompopulation_kdes = [fs.causal_kde1d(x, kdetime, 0.01) for x in hompopulation_spikes]

# compute mean kdes
singlecell_mean = np.mean(singlecell_kdes, axis=0)
hompopulation_mean = np.mean(hompopulation_kdes, axis=0)

# sort by spike train length
hompopulation_spikes = sorted(hompopulation_spikes, key=len, reverse=True)

# plot
fig, ax = plt.subplots()
ax.eventplot(
    hompopulation_spikes,
    lineoffsets=0.3,
    linelengths=0.3,
    colors="black",
    alpha=1,
)
ax.axvline(0, 0, 100, color="black", linestyle="dashed", lw=1)
ax.plot(kdetime, hompopulation_mean, color="green", lw=2)
ax.plot(kdetime, singlecell_mean, color="red", lw=2)
# ax.set_xlim(-0.05, 0.15)
plt.show()
