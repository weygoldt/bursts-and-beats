import matplotlib.pyplot as plt
import numpy as np

import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# make chirping individual (sender)
time, eod_send, _, _ = fs.create_chirp()

# make receiver
eodf_rec = 500
eodf_send = 480
eod_rec = 1 * np.sin(2 * np.pi * eodf_rec * time)
eod_send = 1 * np.sin(2 * np.pi * eodf_send * time)
# make envelope
beat, envelope, envelope_time = fs.beat_envelope(eod_send, eod_rec, 500, eodf_rec, time)

# plot
fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(16 * ps.cm, 12 * ps.cm),
    gridspec_kw={"height_ratios": [1, 1, 2]},
)

plt.subplots_adjust(wspace=0.2)
ax[0].plot(time, eod_send, lw=1, c="k")
ax[1].plot(time, eod_rec, lw=1, c="k")
ax[2].plot(time, beat, lw=1, c="k")
ax[2].plot(
    envelope_time,
    envelope,
    lw=2,
    color="red",
)

ax[0].set_xlim(0.15, 0.25)
ax[2].set_ylim(-2.1, 2.1)

ax[0].set_title(f"Sender: {eodf_send} Hz")
ax[1].set_title(f"Receiver: {eodf_rec} Hz")
ax[2].set_title(f"Amplitude modulation: {eodf_rec-eodf_send} Hz")

# remove axes
for a in ax:
    a.axis("off")

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0, hspace=0.8)

fs.doublesave("figures/beat_without_chirp")
plt.show()
