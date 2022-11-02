import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
from plotstyle import PlotStyle

import functions as fs
from termcolors import TermColor as tc
from plotstyle import PlotStyle

ps = PlotStyle()
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
bl = d.repro_runs("BaselineActivity")
v, t = bl[0].membrane_voltage()
spikes = bl[0].spikes()

def plot_baseline(ax, data, start=0.0, end=1.0, burst=False, single=False):
    bl = data.repro_runs("BaselineActivity")
    v, t = bl[0].membrane_voltage()
    spikes = bl[0].spikes()
    ax.plot(t,v, c=ps.darkblue)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Membrane voltage [mV]')
    ax.scatter(spikes, np.ones_like(spikes)*np.max(v)+1)
    ax.set_xlim(start, end)
    single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(spikes, 0.01, verbose=False)
    if burst==True:
        burst_spikes_fenster = np.where((start < spikes[fs.flatten(burst_spikes)]) &
        (spikes[fs.flatten(burst_spikes)] < end))[0]
        ax.scatter(spikes[fs.flatten(burst_spikes)],
                    np.ones_like(spikes[fs.flatten(burst_spikes)])*np.max(v)+5,
                    label=f"n Bursts: {len(burst_spikes_fenster)}",
                    c="k" )
        ax.legend(bbox_to_anchor=(1.1,1.1), markerscale=1.5)
    if single==True:
        single_spikes_fenster = np.where((start < spikes[single_spikes]) &
        (spikes[single_spikes] < end))[0]
        ax2.scatter(spikes[single_spikes], 
                    np.ones_like(spikes[single_spikes])*np.max(v)+3, 
                    label=f"n Single: {len(single_spikes_fenster)}", 
                    c="blue")
        ax.legend(bbox_to_anchor=(1.1,1.1), markerscale=1.5)
    






#fig1, ax1 = plt.subplots()
#plot_baseline(ax1, d, end=15.0)
#plt.show()

fig2, ax2 = plt.subplots(figsize=(20 * ps.cm, 12 * ps.cm))
plot_baseline(ax2, d, end=10.0, burst=True, single=True)
plt.show()



# zoomed in 

