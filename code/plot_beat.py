import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy import interpolate

import functions as fs
from plotstyle import PlotStyle

s = PlotStyle()


def beat_envelope(sender_eod, receiver_eod, sender_eodf, receiver_eodf, time):
    """
    beat_envelope computes the beat and envelope of the beat from the EODs of two fish.

    The beat is computed using the maxima of the beat betwen the zero crossings of the EOD with the lower EOD frequency.

    Parameters
    ----------
    sender_eod : 1d-array
        EOD of the sender
    receiver_eod : 1d-array
        EOD of the receiver
    sender_eodf : float
        EOD frequency of the sender
    receiver_eodf : float
        EOD frequency of the receiver
    time : 1d-array
        Shared time axis of sender and receiver EOD

    Returns
    -------
    beat : 1d-array
        Beat resulting from the addition of the sender and receiver EOD
    envelope : 1d-array
        Envelope of the beat
    envelope_time : 1d-array
        Time axis of the envelope
    """

    # make beat
    beat = sender_eod + receiver_eod

    # determine which is higher
    lower_eod = receiver_eod if receiver_eodf < sender_eodf else sender_eod

    # rectification
    lower_eod_rect = lower_eod.clip(min=0)

    # find where rectified lower EOD is now 0
    idx = np.arange(len(lower_eod_rect))
    zero_idx = idx[lower_eod_rect != 0]

    # find gaps of continuity in index array
    diffs = np.diff(zero_idx)
    diffs = np.append(diffs, 0)
    zerocrossings = zero_idx[diffs > 1]

    # embed()

    # calculate boundaries
    bounds = [[x, y] for x, y in zip(zerocrossings, zerocrossings[1:])]

    # calculate maxima in non-zero areas
    peaks = []
    for b in bounds:

        # make ranges from boundaries
        b_full = np.arange(b[0], b[1])
        peak = b_full[beat[b_full] == np.max(beat[b_full])][0]
        peaks.append(peak)

    # interpolate between peaks
    interp = interpolate.interp1d(time[peaks], beat[peaks], kind="cubic")
    envelope = interp(time[peaks[0] : peaks[-1]])
    envelope_time = time[peaks[0] : peaks[-1]]

    return beat, envelope, envelope_time


# make receiver
eodf_rec = 500

# make chirping individual (sender)
eodf_send = int(1.05 * eodf_rec)
time, eod_send, _, _ = fs.create_chirp(
    eodf=eodf_send,
    ampl_reduction=0.2,
    chirpsize=200,
    chirptimes=[0.1, 0.185],
)
eod_rec = 0.2 * np.sin(2 * np.pi * eodf_rec * time)

# make envelope
beat, envelope, envelope_time = beat_envelope(
    eod_send, eod_rec, eodf_send, eodf_rec, time
)

# plot
fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(16 * s.cm, 12 * s.cm),
    gridspec_kw={"height_ratios": [1, 1, 2]},
)
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

# ax[0].set_title("Chirping sender")
# ax[1].set_title("Receiver")
# ax[2].set_title("Amplitude modulation")

# remove axes
for a in ax:
    a.axis("off")

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0, hspace=0)

fs.doublesave("../figures/beat")
plt.show()
