import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter
import functions as fs

d = rlx.Dataset("data/2022-10-27-aa-invivo-1.nix")
embed()
exit()
### plot isis
bl1 = d['BaselineActivity_2']

spike_times_bl1 = bl1.spikes()

isis1 = fs.isis(spike_times_bl1)
fig, ax1 = plt.subplots(constrained_layout=True)
fs.plot_isih(ax1, isis1, binwidth=0.009)
#plt.savefig("../figures/isi_histo")
plt.show()
plt.close()

###plot isi corr
corr, lags = fs.isi_serialcorr(isis1)
fig, ax3 = plt.subplots(constrained_layout=True)
plt.plot(lags, corr)
ax3.set_xlabel('Lags k')
ax3.set_ylabel('ISI Correlation p(k)')
#plt.savefig("../figures/isi_corr")
plt.show()
plt.close()

