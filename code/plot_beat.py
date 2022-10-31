import matplotlib.pyplot as plt
import numpy as np

import functions as fs
from plotstyle import PlotStyle

s = PlotStyle()

# make chirping individual (sender)
time, eod_send, _, _ = fs.create_chirp(
    ampl_reduction=0.2, chirpsize=200, chirptimes=[0.10, 0.20, 0.30]
)

# make receiver
eodf_rec = 1.1 * 500
eod_rec = 0.3 * np.sin(2 * np.pi * eodf_rec * time)

# make envelope
beat, envelope, envelope_time = fs.beat_envelope(eod_send, eod_rec, 500, eodf_rec, time)

# plot
fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(24 * s.cm, 12 * s.cm),
    gridspec_kw={"height_ratios": [1, 1, 2]},
)
ax[0].plot(time, eod_send, lw=1, c=s.black)
ax[1].plot(time, eod_rec, lw=1, c=s.black)
ax[2].plot(time, beat, lw=1, c=s.black)
ax[2].plot(
    envelope_time,
    envelope,
    lw=3,
    color=s.green,
)

ax[0].set_xlim(0, 0.4)
ax[2].set_ylim(-2.1, 2.1)

# ax[0].set_title("Chirping sender")
# ax[1].set_title("Receiver")
# ax[2].set_title("Amplitude modulation")

# remove axes
for a in ax:
    a.axis("off")

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0, hspace=0)

fs.doublesave("../figures/beat")
plt.show()
