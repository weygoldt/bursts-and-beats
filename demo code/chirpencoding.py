import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks


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


def chirp_envelope(time, signal, prominence=0.1, distance=1, plot=True):
    # solves the assignment by scipy signal peak detection

    ## rectification
    signal_rec = signal.clip(min=0)

    # peak detection
    peaks, _ = find_peaks(signal_rec, prominence=prominence, distance=distance)

    # cubic spline interpolation
    tck = interpolate.splrep(time[peaks], signal_rec[peaks], s=0)
    envelope = interpolate.splev(time, tck, der=0)

    if plot == True:
        # plotting
        fig, ax = plt.subplots()
        ax.plot(time, signal_rec)
        ax.scatter(time[peaks], signal[peaks])
        ax.plot(time, envelope)
        # ax.set_xlim(0.02, 0.1)
        plt.show()

    return peaks, envelope


def chirp_envelope2(time, signal, plot=True):
    # solves the assignment using zeros that originate from rectification

    ## rectification
    signal_rec = signal.clip(min=0)

    # alternative peaks detection
    zeros = np.where(signal_rec == 0)[0]

    # find gaps of continuity in index array
    lowers = (zeros + 1)[:-1]
    uppers = (zeros - 1)[1:]
    mask = lowers <= uppers
    upperbounds, lowerbounds = uppers[mask], lowers[mask]

    # iterate over signal in bound windows and calc max
    peaks2 = []
    for upper, lower in zip(upperbounds, lowerbounds):
        bounds = np.array(range(lower, upper + 1))
        peak = bounds[signal[bounds] == np.max(signal[bounds])]
        peaks2.append(int(peak))

    # interpolate alternative peaks
    tck = interpolate.splrep(time[peaks2], signal_rec[peaks2], s=0)
    envelope2 = interpolate.splev(time, tck, der=0)

    # plotting alternative peaks
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(time, signal_rec)
        ax.plot(time, envelope2)
        ax.scatter(time[peaks2], signal[peaks2], color="r", marker=".", zorder=100)
        plt.show()

    return peaks2, envelope2


# Goal: From a synthetic beat signal including chirps, compute the beat envelope by

# 1. Rectification: Discard all values below 0
# 2. Peak detection
# 3. Cubic interpolation between peaks

eodf = 500
chirpsize = 100
chirpduration = 0.015
ampl_reduction = 0.2
chirptimes = [0.05, 0.2]
kurtosis = 1.0
duration = 1.0
dt = 0.00001

# simulate data
time, signal, ampl, freq = create_chirp(
    eodf, chirpsize, chirpduration, ampl_reduction, chirptimes, kurtosis, duration, dt
)

eodf_rec = eodf * 2.02
eod = np.sin(2 * np.pi * eodf_rec * time)
mixed = eod + signal * 0.2
plt.plot(time, mixed)
plt.show()

# get envelope
peaks1, envelope1 = chirp_envelope(time, signal)
_, _ = chirp_envelope(time, mixed, distance=int(1 / eodf / dt * 0.9))


# get alternative solution
peaks2, envelope2 = chirp_envelope2(time, signal)
_, _ = chirp_envelope2(time, mixed)
