import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def create_chirp(
    eodf=500,
    chirpsize=100,
    chirpduration=0.015,
    ampl_reduction=0.05,
    chirptimes=[0.05, 0.2],
    kurtosis=1.0,
    duration=1.0,
    dt=0.00001,
):
    """create a fake fish eod that contains chirps at the given times. EOF is a simple sinewave. Chirps are modeled with Gaussian profiles in amplitude reduction and frequency ecxcursion.

    Args:
        eodf (int, optional): The chriping fish's EOD frequency. Defaults to 500 Hz.
        chirpsize (int, optional): the size of the chrip's frequency excursion. Defaults to 100 Hz.
        chirpwidth (float, optional): the duration of the chirp. Defaults to 0.015 s.
        ampl_reduction (float, optional): Amount of amplitude reduction during the chrips. Defaults to 0.05, i.e. 5\%
        chirptimes (list, optional): Times of chirp centers. Defaults to [0.05, 0.2].
        kurtosis (float, optional): The kurtosis of the Gaussian profiles. Defaults to 1.0
        dt (float, optional): the stepsize of the simulation. Defaults to 0.00001 s.

    Returns:
        np.ndarray: the time
        np.ndarray: the eod
        np.ndarray: the amplitude profile
        np.adarray: tha frequency profile
    """
    p = 0.0

    time = np.arange(0.0, duration, dt)
    signal = np.zeros_like(time)
    ampl = np.ones_like(time)
    freq = np.ones_like(time)

    ck = 0
    csig = 0.5 * chirpduration / np.power(2.0 * np.log(10.0), 0.5 / kurtosis)
    for k, t in enumerate(time):
        a = 1.0
        f = eodf

        if ck < len(chirptimes):
            if np.abs(t - chirptimes[ck]) < 2.0 * chirpduration:
                x = t - chirptimes[ck]
                gg = np.exp(-0.5 * np.power((x / csig) ** 2, kurtosis))
                cc = chirpsize * gg

                # g = np.exp( -0.5 * (x/csig)**2 )
                f = chirpsize * gg + eodf
                a *= 1.0 - ampl_reduction * gg
            elif t > chirptimes[ck] + 2.0 * chirpduration:
                ck += 1
        freq[k] = f
        ampl[k] = a
        p += f * dt
        signal[k] = a * np.sin(2 * np.pi * p)

    return time, signal, ampl, freq


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
    if sender_eodf > receiver_eodf:
        lower_eod = receiver_eod
    elif sender_eodf < receiver_eodf:
        lower_eod = sender_eod
    else:
        print("Error: Sender and receiver EODf are the same!")

    # rectification
    lower_eod_rect = lower_eod.clip(min=0)

    # find where rectified lower EOD is now 0
    idx = np.arange(len(lower_eod_rect))
    nonzero_idx = idx[lower_eod_rect != 0]

    # find gaps of continuity in index array
    lowers = (nonzero_idx + 1)[:-1]
    uppers = (nonzero_idx - 1)[1:]
    mask = lowers <= uppers
    upperbounds, lowerbounds = uppers[mask], lowers[mask]

    # calculate maxima in non-zero areas
    peaks = []

    for upper, lower in zip(upperbounds[:-1], lowerbounds[1:]):

        # make ranges from boundaries
        bounds = np.arange(upper, lower)
        peak = bounds[beat[bounds] == np.max(beat[bounds])][0]
        peaks.append(peak)

    # interpolate between peaks
    interp = interpolate.interp1d(time[peaks], beat[peaks], kind="cubic")
    envelope = interp(time[peaks[0] : peaks[-1]])
    envelope_time = time[peaks[0] : peaks[-1]]

    return beat, envelope, envelope_time


# sender eod & chirp parameters
eodf_send = 500
chirpsize = 100
chirpduration = 0.015
ampl_reduction = 0.2
dt = 0.00001

# receiver eod parameters
eodf_rec = eodf_send * 2.05
eod_rec_amp = 1.0

# make sender eod
time, eod_send, _, _ = create_chirp(
    eodf=eodf_send,
    chirpsize=chirpsize,
    chirpduration=chirpduration,
    ampl_reduction=ampl_reduction,
    dt=dt,
)

# make receiver eod
eod_rec = eod_rec_amp * np.sin(2 * np.pi * eodf_rec * time)

beat, envelope, envelope_time = beat_envelope(
    eod_send, eod_rec, eodf_send, eodf_rec, time
)

# plot beat, maxima and envelope
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, constrained_layout=True, sharex=True)
ax1.plot(time, eod_send)
ax2.plot(time, eod_rec)
ax3.plot(time, beat, alpha=0.5)
ax3.plot(envelope_time, envelope, lw=2)

ax3.set_xlim(0.02, 0.15)

plt.show()
