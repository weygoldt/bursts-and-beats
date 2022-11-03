from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
from plotstyle import PlotStyle
import functions as fs

ps = PlotStyle()

d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

### plot isis
bl1 = d['BaselineActivity_4']

spike_times_bl1 = bl1.spikes()

isis1 = fs.isis(spike_times_bl1)

fig1, ax1 = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))
fs.plot_isih(ax1, isis1, binwidth=0.009)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
#plt.savefig("../figures/isi_histo")

plt.show()
#plt.close()

###plot isi corr
corr, lags = fs.isi_serialcorr(isis1)
fig2, ax2 = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))
plt.plot(lags, corr, color=ps.darkblue)
ax2.set_xlabel('Lags k')
ax2.set_ylabel('ISI Correlation p(k)')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
#plt.savefig("../figures/isi_corr")
plt.show()
plt.close()

