import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# get data
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

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

# get spiketimes and voltage trace
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
mean_rates = {}
for key in sorted_trials:
    ind = sorted_trials.get(key)
    rates[key] = []
    mean_rates[key] = []
    rate_per_contrast = []

    for i in ind:
        spikes, _ = fi[i].trace_data("Spikes-1")
        rate = len(spikes) / 0.4
        rate_per_contrast.append(rate)

    rates[key].append(rate_per_contrast)
    mean_rates[key].append(np.mean(rate_per_contrast))

rate_stds = []

for i in rates:
    contrast = i
    std = np.std(rates[i][0])
    rate_stds.append(std)

rate_stds = np.array(rate_stds)

r = np.array([mean_rates[key][0] for key in mean_rates])
c = np.array([key for key in mean_rates])

fig, ax = plt.subplots(figsize=(16 * ps.cm, 12 * ps.cm))
ax.errorbar(c, r, yerr=rate_stds, fmt="-o", color=ps.black, capsize=2)

# ax.plot(c, r, color=ps.black)
# ax.fill_between(c, r - rate_stds, r + rate_stds, color="lightgray", zorder=-10)
# ax.plot([0, 0], [0, 20], ls="dashed", c=ps.black, lw=1)

# remove upper and right axis
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# make axes nicer
ax.set_xticks(range(-30, 35, 10))
ax.set_yticks(range(0, 25, 5))
ax.spines.left.set_bounds((0, 20))
ax.spines.bottom.set_bounds((-30, 30))

# adjust plot margings
plt.subplots_adjust(left=0.1, right=1, top=0.99, bottom=0.1, hspace=0)

# set labels and save to file
ax.set_ylabel("Firing rate [Hz]")
ax.set_xlabel("Contrast")
fs.doublesave("../figures/ficurve")
plt.show()
