
import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

"""
def filter_stimulus(stim):
    contrast = stim.feature_data("rectangle-1_Contrast")
    return contrast > 9.5 and contrast < 12.9

"""

# get data
d = rlx.Dataset("../../data/2022-10-27-aa-invivo-1.nix")
# fi = d.find_stimuli("FIC", filter_stimulus)

#find contrast 
fi = d['FICurve_1']
fi_counts = fi.stimulus_count

contrasts = {}
for count in range(fi_counts):
    contrast = fi[count].feature_data('rectangle-1_Contrast')[0]

    contrasts[f"{count}"] = []
    contrasts[f"{count}"].append(contrast)

sorted_contrasts = {k: v for k, v in sorted(contrasts.items(), key=lambda item:item[1])}


good_fi_spikes, _ = fi[68].trace_data('Spikes-1', before=0.2, after=0.2)
v, t = fi[68].trace_data('LocalEOD-1', before=0.2, after=0.2)

peak_spikes_times = {}

for key in sorted_contrasts.keys():
    key = int(key)
    spikes, _ = fi[key].trace_data('Spikes-1')
    peak_spikes_times[f"{key}"] = []
    for s in spikes:
        if s <= 0.1:
            peak_spikes_times[f"{key}"].append(s)

peak_spikes_rate = {}
for key in peak_spikes_times.keys():
    if peak_spikes_times[f"{key}"] != []:
        peak_spikes_rate[f"{key}"] = []
        rate = len(peak_spikes_times[f"{key}"])/0.1
        peak_spikes_rate[f"{key}"].append(rate)


sorted_trials= {}
spaces = np.linspace(-30,30,10)

for k in sorted_contrasts:

    c = spaces[np.argmin(np.abs(spaces - sorted_contrasts[k][0]))]
    if c in sorted_trials:
        sorted_trials[c].append(int(k))
    else:        
        sorted_trials[c] = [int(k)]       


fig, ax = plt.subplots(constrained_layout=True)
ax.errorbar(c, r, yerr=rate_stds, fmt="-o")
ax.set_ylabel('Firing rate [Hz]')
ax.set_xlabel('Contrasts [%]')
plt.savefig("../figures/fi_curve")
plt.show()
plt.close()
