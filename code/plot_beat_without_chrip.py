import matplotlib.pyplot as plt
import numpy as np

import functions as fs
from plotstyle import PlotStyle

s = PlotStyle()

# make chirping individual (sender)
time, eod_send, _, _ = fs.create_chirp()

# make receiver
eodf_rec = 500
eodf_send = 480
eod_rec = 1 * np.sin(2 * np.pi * eodf_rec * time)
eod_send = 0.4 * np.sin(2 * np.pi * eodf_send * time)
# make envelope
beat, envelope, envelope_time = fs.beat_envelope(eod_send, eod_rec, 500, eodf_rec, time)

# plot
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(16 * s.cm, 12 * s.cm),
    gridspec_kw={"height_ratios": [1, 1, 2]},
)
plt.subplots_adjust(wspace=0.2)
ax1.plot(time, eod_send, lw=1, c="gray")
ax2.plot(time, eod_rec, lw=1, c="gray")
ax3.plot(time, beat, lw=1, c="gray")
ax3.plot(
    envelope_time,
    envelope,
    lw=3,
    color=s.red,
)

ax1.set_xlim(0, 0.3)
ax2.set_ylim(-2.1, 2.1)
ax1.set_ylim(-2.1, 2.1)
ax3.set_ylim(-2.1, 2.1)



ax1.text( 0.0, 1.1, f"Sender: {eodf_send} Hz",transform=ax1.transAxes )
ax2.text( 0.0, 0.99, f"Receiver: {eodf_rec} Hz", transform=ax2.transAxes)
ax2.text(0.0, 1.0, f"Beat Envelope: {eodf_rec-eodf_send} Hz", transform=ax3.transAxes)

# remove axes

ax1.axis("off")

ax2.axis("off")

ax3.axis("off")


plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0.196, hspace=0.196)

fs.doublesave("../figures/beat_without_chirp")
plt.show()
