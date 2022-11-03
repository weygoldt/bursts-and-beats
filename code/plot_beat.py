import matplotlib.pyplot as plt
import numpy as np

import functions as fs
from plotstyle import PlotStyle

s = PlotStyle()
eodf_send = 300
# make chirping individual (sender)
time, eod_send, _, _ = fs.create_chirp(eodf= eodf_send,
    ampl_reduction=0.2, chirpsize=200, chirptimes=[0.1, 0.185]
)

# make receiver

eodf_rec = 300  + 100
eod_rec = 0.2 * np.sin(2 * np.pi * eodf_rec * time)

# make envelope
beat, envelope, envelope_time = fs.beat_envelope(eod_send, eod_rec, eodf_send, eodf_rec, time)

# plot
fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(16 * s.cm, 12 * s.cm),
    gridspec_kw={"height_ratios": [1, 1, 2]},
)
plt.subplots_adjust(wspace=0.2)
ax[0].plot(time, eod_send, lw=1, c="gray")
ax[1].plot(time, eod_rec, lw=1, c="gray")
ax[2].plot(time, beat, lw=1, c="gray")
ax[2].plot(
    envelope_time,
    envelope,
    lw=3,
    color=s.red,
)

ax[0].set_xlim(0, 0.3)
ax[2].set_ylim(-2.1, 2.1)

ax[0].text( 0.0, 1.15, f"Chirping sender: {eodf_send} Hz")
ax[1].text( 0.0, 0.25, f"Receiver: {eodf_rec} Hz",)
ax[2].text(0.0, 1.4, f"Amplitude modulation: {eodf_rec-eodf_send} Hz",)

# remove axes
for a in ax:
    a.axis("off")

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0.196, hspace=0.196)

fs.doublesave("../figures/beat")
plt.show()