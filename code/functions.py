import random

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy import interpolate
from scipy.signal import butter, periodogram, sosfiltfilt
from scipy.stats import gamma, norm

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
        peak = b_full[lower_eod_rect[b_full] == np.max(lower_eod_rect[b_full])][0]
        peaks.append(peak)

    # interpolate between peaks
    interp = interpolate.interp1d(time[peaks], beat[peaks], kind="cubic")
    envelope = interp(time[peaks[0] : peaks[-1]])
    envelope_time = time[peaks[0] : peaks[-1]]

    return beat, envelope, envelope_time


# Interspike interval functions
# Baseline analysis


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
    """
    Detects bursts of spikes where they interspike interval does not cross a certain threshold between

    Parameters
    ----------
    spike_times : _type_
        _description_
    isi_thresh : bool
        _description_
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
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


def plot_baseline(ax, data, start=0.0, end=1.0):
    """Ploting the first recorded Baseline Activity

    Parameters
    ----------
    ax : matplotlib axis
    data : rlx.Dataset
    start : float, optional
        start of the Baseline by default 0.0
    end : float, optional
        end of the Baseline, by default 1.0
    """
    bl = data.repro_runs("BaselineActivity")
    v, t = bl[0].membrane_voltage()
    spikes = bl[0].spikes()
    ax.plot(t,v)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Membrane voltage [mV]')
    ax.scatter(spikes, np.ones_like(spikes)*np.max(v)+1)
    ax.set_xlim(start, end)


# Filters


def bandpass_filter(data, rate, flow, fhigh, order=1):
    sos = butter(order, [flow, fhigh], btype="band", fs=rate, output="sos")
    y = sosfiltfilt(sos, data)
    return y


def lowpass_filter(data, rate, cutoff, order=2):
    """
    lowpass filter

    Parameters
    ----------
    data : 1d array
        data to filter
    rate : float
        sampling rate of the data in Hz
    cutoff : float
        cutoff frequency of the filter in Hz
    order : int, optional
        order of the filter, by default 2

    Returns
    -------
    1d array
        filtered data
    """
    sos = butter(order, cutoff, btype="low", fs=rate, output="sos")
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


def doublesave(title):
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")


# Firing rate estimation


def causal_kde1d(spikes, time, width, shape=2):
    """
    causalkde computes a kernel density estimate using a causal kernel (i.e. exponential or gamma distribution).
    A shape of 1 turns the gamma distribution into an exponential.

    Parameters
    ----------
    spikes : array-like
        spike times
    time : array-like
        sampling time
    width : float
        kernel width
    shape : int, optional
        shape of gamma distribution, by default 1

    Returns
    -------
    rate : array-like
        instantaneous firing rate
    """

    # compute dt
    dt = time[1] - time[0]

    # time on which to compute kernel:
    tmax = 10 * width

    # kernel not wider than time
    if 2 * tmax > time[-1] - time[0]:
        tmax = 0.5 * (time[-1] - time[0])

    # kernel time
    ktime = np.arange(-tmax, tmax, dt)

    # gamma kernel centered in ktime:
    kernel = gamma.pdf(
        x=ktime,
        a=shape,
        loc=0,
        scale=width,
    )

    # indices of spikes in time array:
    indices = np.asarray((spikes - time[0]) / dt, dtype=int)

    # binary spike train:
    brate = np.zeros(len(time))
    brate[indices[(indices >= 0) & (indices < len(time))]] = 1.0

    # convolution with kernel:
    rate = np.convolve(brate, kernel, mode="same")

    return rate


def acausal_kde1d(spikes, time, width):
    """
    causalkde computes a kernel density estimate using a causal kernel (i.e. exponential or gamma distribution).
    A shape of 1 turns the gamma distribution into an exponential.

    Parameters
    ----------
    spikes : array-like
        spike times
    time : array-like
        sampling time
    width : float
        kernel width
    shape : int, optional
        shape of gamma distribution, by default 1

    Returns
    -------
    rate : array-like
        instantaneous firing rate
    """

    # compute dt
    dt = time[1] - time[0]

    # time on which to compute kernel:
    tmax = 10 * width

    # kernel not wider than time
    if 2 * tmax > time[-1] - time[0]:
        tmax = 0.5 * (time[-1] - time[0])

    # kernel time
    ktime = np.arange(-tmax, tmax, dt)

    # gamma kernel centered in ktime:
    kernel = norm.pdf(
        x=ktime,
        loc=0,
        scale=width,
    )

    # indices of spikes in time array:
    indices = np.asarray((spikes - time[0]) / dt, dtype=int)

    # binary spike train:
    brate = np.zeros(len(time))
    brate[indices[(indices >= 0) & (indices < len(time))]] = 1.0

    # convolution with kernel:
    rate = np.convolve(brate, kernel, mode="same")

    return rate


# Data access and sorting


def singlecell_cts(data):
    """
    singlecell_cts extracts the spikes centered around the chirp stimulus
    (chirp-triggered spikes) onset for every single chirp stimulus.

    Parameters
    ----------
    data : rlxnix dataset
        Relacs dataset imported with rlxnix

    Returns
    -------
    spike_t : array of arrays
        an array including an array for every ct-spiketrain
    c_time : array
        the time centered around the chirp
    """

    # collect chirp-centered spike times here
    spike_t = []

    # padding around chirp
    before_t = 0.15
    after_t = 0.15

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = data[repro]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)
            spikes = chirps.spikes(i)
            chirp_times = chirps.chirp_times[0][i]

            # compute number of indices before and after chirp to include
            dt = time[1] - time[0]
            before_indices = np.round(before_t / dt)
            after_indices = np.round(after_t / dt)

            for c in chirp_times:

                # where is chirp on time vector?
                c_index = find_closest(time, c)

                # make index vector centered around chirp
                indices = np.arange(
                    c_index - before_indices, c_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    c_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    continue

                tmin = np.min(c_time)
                tmax = np.max(c_time)

                # get spike times in this range
                c_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

                # get spike indices on c_time vector
                c_spike_indices = [find_closest(c_time, x) for x in c_spikes]

                # make new centered time array
                c_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

                # extract spike timestamps from centered time
                c_spikes_centered = c_time[c_spike_indices]

                # append centered spike times to list
                spike_t.append(c_spikes_centered)

    return spike_t, c_time


def hompopulation_cts(data):
    """
    hompopulation_cts extracts the spikes centered around the chirp stimulus
    (chirp-triggered spikes) onset for every single chirp stimulus and groups
    them per trial. This simulates a population response for a homogenous
    population of this cell.

    Parameters
    ----------
    data : rlxnix dataset
        Relacs dataset imported with rlxnix

    Returns
    -------
    spike_t : array of arrays
        an array including an array for every ct-spiketrain
    c_time : array
        the time centered around the chirp
    """

    # collect chirp-centered spike times here
    spike_t = []

    # padding around chirp
    before_t = 0.15
    after_t = 0.15

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = data[repro]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)
            spikes = chirps.spikes(i)
            chirp_times = chirps.chirp_times[0][i]

            # compute number of indices before and after chirp to include
            dt = time[1] - time[0]
            before_indices = np.round(before_t / dt)
            after_indices = np.round(after_t / dt)

            # collect spikes for this trial here
            spikelist = []

            for c in chirp_times:

                # where is chirp on time vector?
                c_index = find_closest(time, c)

                # make index vector centered around chirp
                indices = np.arange(
                    c_index - before_indices, c_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    c_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    continue

                tmin = np.min(c_time)
                tmax = np.max(c_time)

                # get spike times in this range
                c_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

                # get spike indices on c_time vector
                c_spike_indices = [find_closest(c_time, x) for x in c_spikes]

                # make new centered time array
                c_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

                # extract spike timestamps from centered time
                c_spikes_centered = c_time[c_spike_indices]

                # append centered spike times to list
                spikelist.append(c_spikes_centered)

            # flatten spike list to simulate activity of hom population
            spikelist_flat = flatten(spikelist)

            # save to spike times list
            spike_t.append(spikelist_flat)

    return spike_t, c_time


def singlecell_bts(data):

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # collect beat-centered spike times here
    spike_t = []

    # before and after padding
    before_t = 0.15
    after_t = 0.15

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = data[repro]
        chirp_duration = chirps.metadata["RePro-Info"]["settings"]["chirpwidth"][0][0]
        fish_eodf = data.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)

            if np.max(time) < 1:
                print(tc.err("Trial too short, skipping"))
                continue

            dt = time[1] - time[0]
            rate = 1 / dt
            spikes = chirps.spikes(i)
            chirp_times = chirps.chirp_times[0][i]
            nchirps = len(chirp_times)
            stim, stimtime = chirps.stimulus_output(i)

            # fish data
            fish_eod, fish_eodtime = chirps.eod(i)

            # signal data
            stim_eod, stim_eodtime = chirps.stimulus_output(i)
            stim_eodf = rel_to_eods(fish_eodf, chirps.relative_eodf)

            # make envelope
            beat, envelope, envelope_time = beat_envelope(
                stim_eod, fish_eod, stim_eodf, fish_eodf, time
            )

            # Find chirp areas -------------------------------------------------------------

            # find index of chirp
            chirp_indices = [find_closest(time, c) for c in chirp_times]

            # make window around chirp according to chirp duration
            chirp_windows = [
                [c - chirp_duration / 2, c + chirp_duration / 2] for c in chirp_times
            ]

            # find indices of chirp window start and stop
            chirp_window_indices = [
                [find_closest(time, x[0]), find_closest(time, x[1])]
                for x in chirp_windows
            ]

            # convert indices to NON-chirp window start stop
            nonchirp_indices = []
            for i, w in enumerate(chirp_window_indices):

                # start first window with 0
                if w[0] == chirp_window_indices[0][0]:
                    nonchirp_indices.append([0, w[0]])

                # end last window with len(time)
                elif w[1] == chirp_window_indices[-1][1]:
                    nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])
                    nonchirp_indices.append([w[1], len(time)])

                # take last of previous and first of current for middle windows
                else:
                    nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])

            # split envelope in non chirp periods
            env_split = [envelope[x[0] : x[1]] for x in nonchirp_indices]
            env_time = [envelope_time[x[0] : x[1]] for x in nonchirp_indices]

            # compute peak timestamps by sine approximation for each envelope snippet
            try:
                env_peaks = flatten(
                    [
                        t[envelope_peaks(env, t, rate)]
                        for env, t in zip(env_split, env_time)
                    ]
                )
            except:
                embed()

            # convert peak timestamps to indices on whole time array
            beat_peaks = [find_closest(time, x) for x in env_peaks]

            # draw random beat peaks
            selected_beats = random.sample(beat_peaks, nchirps)

            # Center the time at the beat peak ---------------------------------------------

            # compute number of indices before and after chirp to include
            dt = time[1] - time[0]
            before_indices = np.round(before_t / dt)
            after_indices = np.round(after_t / dt)

            for sb in time[selected_beats]:

                # where is index on the time vector?
                b_index = find_closest(time, sb)

                # make index vector centered around beat
                indices = np.arange(
                    b_index - before_indices, b_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    b_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    # embed()
                    continue

                tmin = np.min(b_time)
                tmax = np.max(b_time)

                # get spike times in this range
                b_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

                # get spike indices on b_time vector
                b_spike_indices = [find_closest(b_time, x) for x in b_spikes]

                # make new centered time array
                b_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

                # extract spike timestamps from centered time
                b_spikes_centered = b_time[b_spike_indices]

                # append centered spike times to list
                spike_t.append(b_spikes_centered)

    return spike_t, b_time


def hompopulation_bts(data):

    # collect beat-centered spike times here
    spike_t = []

    # padding around beat
    before_t = 0.15
    after_t = 0.15

    # find all chirp repros
    chirp_repros = [i for i in data.repros if "Chirps" in i]

    # go through all chirp repros
    for repro in chirp_repros:

        # get chirps from each repro
        chirps = data[repro]
        chirp_duration = chirps.metadata["RePro-Info"]["settings"]["chirpwidth"][0][0]
        fish_eodf = data.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]

        for i in range(chirps.stimulus_count):

            # get data
            _, time = chirps.membrane_voltage(i)

            if np.max(time) < 1:
                print(tc.err("Trial too short, skipping"))
                continue

            dt = time[1] - time[0]
            rate = 1 / dt
            spikes = chirps.spikes(i)
            chirp_times = chirps.chirp_times[0][i]
            nchirps = len(chirp_times)
            stim, stimtime = chirps.stimulus_output(i)

            # fish data
            fish_eod, fish_eodtime = chirps.eod(i)

            # signal data
            stim_eod, stim_eodtime = chirps.stimulus_output(i)
            stim_eodf = rel_to_eods(fish_eodf, chirps.relative_eodf)

            # make envelope
            beat, envelope, envelope_time = beat_envelope(
                stim_eod, fish_eod, stim_eodf, fish_eodf, time
            )

            # Find chirp areas -------------------------------------------------------------

            # find index of chirp
            chirp_indices = [find_closest(time, c) for c in chirp_times]

            # make window around chirp according to chirp duration
            chirp_windows = [
                [c - chirp_duration / 2, c + chirp_duration / 2] for c in chirp_times
            ]

            # find indices of chirp window start and stop
            chirp_window_indices = [
                [find_closest(time, x[0]), find_closest(time, x[1])]
                for x in chirp_windows
            ]

            # convert indices to NON-chirp window start stop
            nonchirp_indices = []
            for i, w in enumerate(chirp_window_indices):

                # start first window with 0
                if w[0] == chirp_window_indices[0][0]:
                    nonchirp_indices.append([0, w[0]])

                # end last window with len(time)
                elif w[1] == chirp_window_indices[-1][1]:
                    nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])
                    nonchirp_indices.append([w[1], len(time)])

                # take last of previous and first of current for middle windows
                else:
                    nonchirp_indices.append([chirp_window_indices[i - 1][1], w[0]])

            # split envelope in non chirp periods
            env_split = [envelope[x[0] : x[1]] for x in nonchirp_indices]
            env_time = [envelope_time[x[0] : x[1]] for x in nonchirp_indices]

            # compute peak timestamps by sine approximation for each envelope snippet
            try:
                env_peaks = flatten(
                    [
                        t[envelope_peaks(env, t, rate)]
                        for env, t in zip(env_split, env_time)
                    ]
                )
            except:
                embed()

            # convert peak timestamps to indices on whole time array
            beat_peaks = [find_closest(time, x) for x in env_peaks]

            # draw random beat peaks
            selected_beats = random.sample(beat_peaks, nchirps)

            # Center the time at the beat peak ---------------------------------------------

            # compute number of indices before and after chirp to include
            dt = time[1] - time[0]
            before_indices = np.round(before_t / dt)
            after_indices = np.round(after_t / dt)

            # collect beat triggered spikes for this trial here
            spikelist = []

            for sb in time[selected_beats]:

                # where is index on the time vector?
                b_index = find_closest(time, sb)

                # make index vector centered around beat
                indices = np.arange(
                    b_index - before_indices, b_index + after_indices, dtype=int
                )

                # get max t and min t
                try:
                    b_time = time[indices]
                except:
                    print(tc.warn(f"Trial {i} Repro {repro} skipped, not enough data!"))
                    print(f"max index: {np.max(indices)}")
                    print(f"time length: {len(time)}")
                    continue

                tmin = np.min(b_time)
                tmax = np.max(b_time)

                # get spike times in this range
                b_spikes = spikes[(spikes > tmin) & (spikes < tmax)]

                # get spike indices on b_time vector
                b_spike_indices = [find_closest(b_time, x) for x in b_spikes]

                # make new centered time array
                b_time = np.arange(-before_indices * dt, (after_indices + 1) * dt, dt)

                # extract spike timestamps from centered time
                b_spikes_centered = b_time[b_spike_indices]

                # append centered spike times to list
                spikelist.append(b_spikes_centered)

            # flatten spike list to simulate activity of hom population
            spikelist_flat = flatten(spikelist)

            # save to spike times list
            spike_t.append(spikelist_flat)

    return spike_t, b_time


def sort_reodfs(data):
    """Sorting of the relative EODs of chirps data.

    Parameters
    ----------
    data : rlx.Dataset nix file
        Dataset with different EODs for chirp data
    Returns
    -------
    dic
        Dictionary with the relative EODs as keys, Items are the name of the trace
    """
    r_eodf = []
    for chirp in data.repro_runs("Chirps"):
        r_eodf.append(chirp.relative_eodf)

    r_eodf_arr = np.array(r_eodf)
    r_eodf_arr_uniq = np.unique(r_eodf_arr)

    r_eodf_dict = {}

    for unique_r_eodf in r_eodf_arr_uniq:
        r_eodf_dict[f"{unique_r_eodf}"] = []
        for r in range(len(r_eodf)):
            chirps = data.repro_runs("Chirps")[r]
            if unique_r_eodf == r_eodf[r]:
                r_eodf_dict[f"{unique_r_eodf}"].append(chirps.name)

    return r_eodf_dict


# small tools


def rel_to_eods(fish_eod, rel):
    """Converts the relative EOD to the fish to the stim EOD"""
    return (rel - 1) * fish_eod + fish_eod


def eods_to_rel(fish_eod, stim_eod):
    """Converts the stimulus EOD to the relative EOD to the fish"""
    return ((stim_eod - fish_eod) / fish_eod) + 1


def coustomnorm(data):
    """
    coustomnorm normalizes an array between -1 and 1.

    Parameters
    ----------
    data : array
        The input data

    Returns
    -------
    array
        The normalized array
    """
    return 2 * ((data - min(data)) / (max(data) - min(data))) - 1


def euclidean(u, v):
    """
    euclidean computes the Euclidean distance between two arrays.

    Parameters
    ----------
    u : array
        The first array
    v : array
        The second array

    Returns
    -------
    float
        The Euclidean distance between the arrays.
    """
    return np.sqrt(np.sum(np.abs(u - v) ** 2))


def envelope_peaks(envelope, time, rate):
    """
    envelope_peaks computes the peaks of a beat envelope by approximating the envelope with a sine of the same frequency and phase of the smalles Euclidean distance to the envelope.

    Parameters
    ----------
    envelope : array-like
        The beat envelope.
    time : array-like
        The time array corresponding to the envelope.
    rate :  float
        Sampling rate of data and envelope

    Returns
    -------
    peaks : array
        Indices for the peaks on the envelope.
    """

    # calculate envelope power spectrum
    f, p = periodogram(envelope, rate)

    # get frequency of maximum power (i.e. envelope frequency)
    f_env = f[p == np.max(p)][0]

    # make many sines in envelope frequencies and different phases
    phases = np.arange(0, 2.0 * np.pi, 0.1)  # phases
    sines = np.array([np.sin(2.0 * np.pi * f_env * time + p) for p in phases])

    # compute euclidean distance from envelope to all sines
    dist = [euclidean(envelope, x) for x in sines]

    # get sine at phase where euclidean distance is lowest
    env_approx = sines[phases == phases[dist == np.min(dist)][0]][0]

    # rectify sine approximated envelope
    env_approx[env_approx < 0] = 0

    # find where rectified sine is now 0
    idx = np.arange(len(env_approx))
    zero_idx = idx[env_approx != 0]

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
        peak = b_full[env_approx[b_full] == np.max(env_approx[b_full])][0]
        peaks.append(peak)

    return peaks
