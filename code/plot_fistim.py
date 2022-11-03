import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
from scipy import interpolate
import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# get data
d = rlx.Dataset("data/2022-10-27-aa-invivo-1.nix")

# find contrast
fi = d["FICurve_1"]

eodf = fi.trace_data('EOD')[0]
eodx = fi.trace_data('EOD')[1]

localeodf = fi.trace_data('LocalEOD-1')[0]
localeodx = fi.trace_data('LocalEOD-1')[1]
beat = localeodf
# compute envelope

# rectification
lower_eod_rect = eodf.clip(max=0)

# find where rectified lower EOD is now 0
idx = np.arange(len(lower_eod_rect))
zero_idx = idx[lower_eod_rect != 0]

# find gaps of continuity in index array
diffs = np.diff(zero_idx)
diffs = np.append(diffs, 0)
zerocrossings = zero_idx[diffs > 1]

# calculate boundaries
bounds = [[x, y] for x, y in zip(zerocrossings, zerocrossings[1:])]

# calculate maxima in non-zero areas
peaks = []
for b in bounds:

    # make ranges from boundaries
    b_full = np.arange(b[0], b[1])
    peak = b_full[beat[b_full] == np.min(beat[b_full])][0]
    peaks.append(peak)

# fix noisy peaks
peaks = np.array(peaks)[beat[peaks] < -0.4]

# interpolate between peaks
interp = interpolate.interp1d(
    localeodx[peaks], beat[peaks], kind="cubic", fill_value="extrapolate"
)
envelope = interp(localeodx)


fig, ax = plt.subplots()

#plt.plot(localeodx, localeodf)
plt.plot(localeodx, envelope * -1, color = ps.darkblue)

# remove upper and right axis
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.xlim((0, 16))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Stimulus envelope')
fs.doublesave("figures/fistim")
plt.show()

