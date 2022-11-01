import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

import functions as fs
from termcolors import TermColor as tc


d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")


bl = d["BaselineActivity_4"]
#spikes = bl["Spikes"]

v, t = bl.membrane_voltage()
spikes = bl.spikes()
fig, ax = plt.subplots()
ax.plot(t,v)
ax.set_xlabel('time [s]')
ax.set_ylabel('membrane voltage [mV]')
plt.plot(t,v)
plt.show()

fig1, ax1 = plt.subplots()
ax1.plot(t,v)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('membrane voltage [mV]')
ax1.set_xlim(0, 1)
plt.plot(t,v)
plt.show()

single_spikes, burst_spikes, burst_start_stop = fs.burst_detector(spikes, 0.01)


fig2, ax2 = plt.subplots()
ax2.plot(t,v, c="orange")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Membrane Voltage [mV]')
#ax2.text(0.8, 1.0, f"n Bursts: {len(burst_spikes)}", transform=ax.transAxes)
#ax2.text(0.8, 0.95, f"n Single: {len(single_spikes)}", transform=ax.transAxes)
ax2.scatter(spikes[single_spikes], np.ones_like(spikes[single_spikes])*np.max(v)+1, marker='|',s=20,  label=f"n Bursts: {len(burst_spikes)}" , c="blue")
ax2.scatter(spikes[fs.flatten(burst_spikes)], np.ones_like(spikes[fs.flatten(burst_spikes)],)*np.max(v)+2, s=20,marker='|',label=f"n Single: {len(single_spikes)}", c="k" )
ax2.legend(bbox_to_anchor=(1.1,1.1))
plt.show()



