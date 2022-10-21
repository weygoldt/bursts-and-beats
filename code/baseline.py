import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

# PLAN
#

# import data
d = rlx.Dataset("../data/2022-10-20-ab-invivo-1.nix")

# extract baseline
bl = d["BaselineActivity_2"]

# extract voltage and time
spike_times = bl.spikes()
v, t = bl.membrane_voltage()

# plot
fig, ax = plt.subplots()
ax.plot(t, v)
ax.scatter(spike_times, np.ones_like(spike_times) * np.max(v), marker="|")
plt.show()
