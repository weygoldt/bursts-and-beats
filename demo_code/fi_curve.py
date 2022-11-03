
import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

<<<<<<< HEAD
import functions as fs
from termcolors import TermColor as tc

=======
>>>>>>> Kathi
"""
def filter_stimulus(stim):
    contrast = stim.feature_data("rectangle-1_Contrast")
    return contrast > 9.5 and contrast < 12.9

"""

# get data
d = rlx.Dataset("data/2022-10-27-aa-invivo-1.nix")
# fi = d.find_stimuli("FIC", filter_stimulus)

# find contrast
fi = d["FICurve_1"]
fi_counts = fi.stimulus_count

contrasts = {}
for count in range(fi_counts):
    contrast = fi[count].feature_data("rectangle-1_Contrast")[0]

    contrasts[f"{count}"] = []
    contrasts[f"{count}"].append(contrast)

sorted_contrasts = {
    k: v for k, v in sorted(contrasts.items(), key=lambda item: item[1])
}


good_fi_spikes, _ = fi[68].trace_data("Spikes-1", before=0.2, after=0.2)
v, t = fi[68].trace_data("LocalEOD-1", before=0.2, after=0.2)

peak_spikes_times = {}

for key in sorted_contrasts.keys():
    key = int(key)
    spikes, _ = fi[key].trace_data("Spikes-1")
    peak_spikes_times[f"{key}"] = []
    for s in spikes:
        if s <= 0.1:
            peak_spikes_times[f"{key}"].append(s)

peak_spikes_rate = {}
for key in peak_spikes_times.keys():
    if peak_spikes_times[f"{key}"] != []:
        peak_spikes_rate[f"{key}"] = []
        rate = len(peak_spikes_times[f"{key}"]) / 0.1
        peak_spikes_rate[f"{key}"].append(rate)


sorted_trials = {}
spaces = np.linspace(-30, 30, 10)

for k in sorted_contrasts:

    c = spaces[np.argmin(np.abs(spaces - sorted_contrasts[k][0]))]
    if c in sorted_trials:
        sorted_trials[c].append(int(k))
    else:
        sorted_trials[c] = [int(k)]

rates = {}
<<<<<<< HEAD
for key in sorted_trials:
    ind = sorted_trials.get(key)
    rate_per_contrast = []
    rates[key] = []
    for i in ind:
        spikes, _ = fi[i].trace_data("Spikes-1")
        rate = len(spikes) / 0.4
        rate_per_contrast.append(rate)

    rates[key].append(np.mean(rate_per_contrast))
    rates[key].append(np.std(rate_per_contrast))
=======
mean_rates = {}
for key in sorted_trials:
    ind = sorted_trials.get(key)
    rates[key] = []
    mean_rates[key] = []
    rate_per_contrast = []
    for i in ind:
        spikes, _ = fi[i].trace_data('Spikes-1')
        rate = len(spikes) / 0.4
        rate_per_contrast.append(rate)
    
    rates[key].append(rate_per_contrast)
    mean_rates[key].append(np.mean(rate_per_contrast))
>>>>>>> Kathi

rate_stds = []
for i in rates:
    contrast = i
    std = np.std(rates[i][0])
    rate_stds.append(std)

<<<<<<< HEAD
r = []
c = []
std = []
for key in rates:
    c.append(key)
    r.append(rates[key][0])
    std.append(rates[key][1])

fig, ax = plt.subplots()
ax.errorbar(c, r, yerr=std)
plt.show()


# firering rate and std for each space !!
=======
r = [mean_rates[key][0] for key in mean_rates]
c = [key for key in mean_rates]

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(c, r, yerr=rate_stds, fmt="-o")
ax.set_ylabel('Firing rate [Hz]')
ax.set_xlabel('Contrasts')
plt.show()
>>>>>>> Kathi
