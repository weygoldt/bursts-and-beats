import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy.interpolate import interp1d
from scipy.signal import medfilt

import functions as fs

# load data
d = rlx.Dataset("../data/2022-10-20-ab-invivo-1.nix")

# extract baseline
bl = d["BaselineActivity_2"]

# extract voltage and time
spike_times = bl.spikes()
v, t = bl.membrane_voltage()

# filter signal to cancel slow changes in membrane potential
rate = 1 / (t[1] - t[0])  # in Hz
nqst = rate / 2  # Niquist frequency
v_filt = fs.bandpass_filter(v, rate, 0.2, nqst * 0.99)

# find indices of spike times in time array
spike_indices = [fs.find_closest(t, x) for x in spike_times]

# compute voltage at spike peaks
spike_v = v_filt[spike_indices]

# compute spike amplitudes
v_mean = np.mean(v_filt)
spike_amp = spike_v - v_mean

# scale dynamic threshold and apply median filter
thresh_amp = spike_amp * 0.1
thresh = v_mean + thresh_amp
thresh = medfilt(thresh, 51)

# interpolate and pad threshold
interp = interp1d(t[spike_indices], thresh, kind="cubic", fill_value="extrapolate")
thresh = interp(t)

plt.plot(t, v_filt)
plt.plot(t, thresh)
plt.scatter(t[spike_indices], spike_v, color="red")
plt.scatter(spike_times, np.ones_like(spike_times) * 30)
plt.show()
