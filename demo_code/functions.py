import numpy as np
from scipy import interpolate
from scipy.signal import butter, sosfiltfilt

from termcolors import TermColor as tc

# Data simulation


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


# Data manipulation


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
    zero_idx = idx[lower_eod_rect != 0]

    # find gaps of continuity in index array
    lowers = (zero_idx + 1)[:-1]
    uppers = (zero_idx - 1)[1:]
    mask = lowers <= uppers
    upperbounds, lowerbounds = uppers[mask], lowers[mask]

    # calculate maxima in non-zero areas
    peaks = []
    for upper, lower in zip(upperbounds[0:-2], lowerbounds[1:-1]):

        # make ranges from boundaries
        bounds = np.arange(upper, lower)
        peak = bounds[beat[bounds] == np.max(beat[bounds])][0]
        peaks.append(peak)

    # interpolate between peaks
    interp = interpolate.interp1d(time[peaks], beat[peaks], kind="cubic")
    envelope = interp(time[peaks[0] : peaks[-1]])
    envelope_time = time[peaks[0] : peaks[-1]]

    return beat, envelope, envelope_time


# Interspike interval functions


def isis(spike_times):
    """
    Compute interspike intervals of spike times per recording trial.

    Parameters
    ----------
    spike_times : array-like of arrays
        A list/array of trials containing spike times

    Returns
    -------
    isiarray : array of float
        Interspike intervals
    """

    isiarray = []
    for times in spike_times:
        difftimes = np.diff(times)
        isiarray.append(difftimes)

    return isiarray


def isih(isis, bin_width):
    """
    isih computes the probability density function of the interspike intervals.

    Parameters
    ----------
    isis : bool
        _description_
    bin_width : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    bins = np.arange(0.0, np.max(isis) + bin_width, bin_width)
    counts, edges = np.histogram(isis, bins)
    centers = edges[:-1] + 0.5 * bin_width
    pdf = counts / np.sum(counts) / bin_width

    return pdf, centers


def plot_isih(ax, isis, binwidth):
    """
    Plot the interspike interval histogram.

    Parameters
    ----------
    ax : matplotlib axis
    isis : 1d-array of floats
        The interspike intervals
    binwidth : float
        Bin width to be used for the histogram
    """

    pdf, centers = isih(isis, binwidth)

    # compute histogram
    misi = np.mean(isis)

    # basic statistics
    sdisi = np.std(isis)
    cv = sdisi / misi
    ax.bar(centers * 1000, pdf, width=binwidth * 1000)  # plot histogram with ISIs in ms

    ax.set_xlabel("Interspike interval [ms]")
    ax.set_ylabel("p(ISI) [1/s]")

    # annotate plot with relative coordinates (0-1, transform argument):
    # f-string to put variables values directly into the string (within {}).
    # r-string: no need to escape backslashes.
    # Matplotlib math mode enclosed in '$' supports LaTeX style typesetting.
    # In math-mode greek letters are available by their name with backslash.
    # Subscripts are introduced by '_' and are enclosed in curly brackets.
    # Since we are in an f-string we need to double the curly brackets.

    ax.text(0.8, 0.9, rf"$\mu_{{ISI}}={misi*1000:.1f}$ms", transform=ax.transAxes)
    ax.text(0.8, 0.8, rf"$\sigma_{{ISI}}={sdisi*1000:.1f}$ms", transform=ax.transAxes)
    ax.text(0.8, 0.7, rf"CV={cv:.2f}", transform=ax.transAxes)


def isi_serialcorr(isis, max_lag=10):
    """
    Serial correlations of interspike intervals

    Parameters
    ----------
    isis : 1d-array of floats
        Interspike intervals
    max_lag : int, optional
        Maximum lag, by default 10

    Returns
    -------
    isicorr : array of floats
        Interspike interval correlations
    lags : array of integers
        Lags of interval correlations
    """

    lags = np.arange(max_lag + 1)
    isicorr = np.zeros(len(lags))
    nisis = len(isis)
    for k in range(len(lags)):  # for each lag
        lag = lags[k]
        if nisis > lag + 10:
            # ensure "enough" data
            isicorr[k] = np.corrcoef(isis[: nisis - lag], isis[lag:])[0, 1]
    return isicorr, lags


def burst_detector(spike_times, isi_thresh, verbose=True):

    # compute interspike intervals
    isi = isis([spike_times])[0]

    # find indices of spike times in time array
    spike_indices = np.arange(len(spike_times))

    # find spikes where at least one sourrounding isi is lower than the threshold
    burst_spikes = []
    single_spikes = []
    burst = False
    switch = False

    for spike in spike_indices:

        # first spike
        if spike == spike_indices[0]:
            spike_isi = isi[0]

            # test if greater than thresh
            if spike_isi < isi_thresh and burst is False:
                burst = True
                burst_list = []
                burst_list.append(spike)
            else:
                burst = False

        # last spike
        elif spike == spike_indices[-1]:
            spike_isi = isi[-1]

            # test if greater than thresh
            if spike_isi < isi_thresh and burst is True:
                burst_list.append(spike)
            else:
                burst = False

        # middle spikes
        else:
            spike_isi = isi[spike - 1 : spike + 1]

            # test if greater than thresh
            if (
                (spike_isi[0] < isi_thresh) or (spike_isi[1] < isi_thresh)
            ) and burst is True:

                # the burst stops if the next ISI is greater
                if spike_isi[1] > isi_thresh:
                    switch = True

                burst_list.append(spike)

            elif (
                (spike_isi[0] < isi_thresh) or (spike_isi[1] < isi_thresh)
            ) and burst is False:
                burst = True
                burst_list = []
                burst_list.append(spike)
            else:
                burst = False

        if switch:
            burst_spikes.append(burst_list)
            burst_list = []

        if burst is False:
            single_spikes.append(spike)

        switch = False

    # convert to numpy arrays
    burst_spikes = np.array(burst_spikes, dtype=object)
    single_spikes = np.array(single_spikes)

    # compute start and stop of each burst
    burst_start_stop = []
    for burst in burst_spikes:
        burst_start_stop.append([burst[0], burst[-1]])
    burst_start_stop = np.array(burst_start_stop)

    if verbose is True:
        print(f"{tc.succ('Burst threshold: ')}{isi_thresh} seconds interspike interval")
        print(
            f"{tc.succ('Burst spikes: ')}{len(flatten(burst_spikes))} spikes in {len(burst_spikes)} bursts"
        )
        print(f"{tc.succ('Single spikes: ')}{len(single_spikes)} spikes")

    return single_spikes, burst_spikes, burst_start_stop


# Filters


def bandpass_filter(data, rate, flow, fhigh, order=1):
    sos = butter(order, [flow, fhigh], btype="band", fs=rate, output="sos")
    y = sosfiltfilt(sos, data)
    return y


# Other


def find_closest(array, target):
    """Takes an array and a target and returns an index for a value of the array that matches the target most closely.

    Could also work with multiple targets and may not work for unsorted arrays, i.e. where a value occurs multiple times. Primarily used for time vectors.

    Parameters
    ----------
    array : array, required
        The array to search in.
    target : float, required
        The number that needs to be found in the array.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target
    return idx


def flatten(l):
    """
    Flattens a list / array of lists.

    Parameters
    ----------
    l : array or list of lists
        The list to be flattened

    Returns
    -------
    list
        The flattened list
    """
    return [item for sublist in l for item in sublist]
