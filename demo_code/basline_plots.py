import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

import functions as fs
from termcolors import TermColor as tc


d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

def plot_baseline(ax, data, start=0.0, end=1.0):
    bl = data.repro_runs("BaselineActivity")
    v, t = bl[0].membrane_voltage()
    spikes = bl[0].spikes()
    ax.plot(t,v)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('membrane voltage [mV]')
    ax.scatter(spikes, np.ones_like(spikes)*np.max(v)+1)
    ax.set_xlim(start, end)



fig1, ax1 = plt.subplots()
plot_baseline(ax1, d, end=15.0)
plt.show()

single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(spikes, 0.01)

# burst spikes 
fig2, ax2 = plt.subplots()
ax2.plot(t,v, c="orange")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Membrane Voltage [mV]')
#ax2.text(0.8, 1.0, f"n Bursts: {len(burst_spikes)}", transform=ax.transAxes)
#ax2.text(0.8, 0.95, f"n Single: {len(single_spikes)}", transform=ax.transAxes)
ax2.scatter(spikes[single_spikes], np.ones_like(spikes[single_spikes])*np.max(v)+1, marker='|',s=20,  label=f"n Bursts: {len(burst_spikes)}" , c="blue")
ax2.scatter(spikes[fs.flatten(burst_spikes)], np.ones_like(spikes[fs.flatten(burst_spikes)],)*np.max(v)+2, s=20,marker='|',label=f"n Single: {len(single_spikes)}", c="k" )
ax2.legend(bbox_to_anchor=(1.1,1.1), fontsize=10, markerscale=6)

plt.show()

# zoomed in 

fig3, ax3 = plt.subplots()
ax3.plot(t,v, c="orange")
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Membrane Voltage [mV]')
#ax3.text(0.8, 1.0, f"n Bursts: {len(burst_spikes)}", transform=ax.transAxes)
#ax3.text(0.8, 0.95, f"n Single: {len(single_spikes)}", transform=ax.transAxes)
ax3.scatter(spikes[single_spikes], np.ones_like(spikes[single_spikes])*np.max(v)+1, marker='|',linewidths=2, label=f"n Bursts: {len(burst_spikes)}" , c="blue")
ax3.scatter(spikes[fs.flatten(burst_spikes)], np.ones_like(spikes[fs.flatten(burst_spikes)],)*np.max(v)+2, linewidths=2,marker='|',label=f"n Single: {len(single_spikes)}", c="k" )
ax3.legend(bbox_to_anchor=(1.1,1.1), markerscale=4)
ax3.set_xlim(0, 1)
plt.show()


