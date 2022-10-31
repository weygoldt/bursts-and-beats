import enum
from tracemalloc import start

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from scipy import interpolate

import functions as fs
from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()

# get data
data = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
sams = [i for i in data.repros if "SAM" in i]
sams = sams[2:]
sam = sams[2]
samd = data[sam]
spikes = samd.trace_data("Spikes-1")[0]
beat, time = samd.trace_data("LocalEOD-1")
dt = time[1] - time[0]
eod, time = samd.trace_data("EOD")
pause = 5
stim_dur = 30

# start at 5 seconds and cut next 30 seconds into period bins
startindex = fs.find_closest(time, pause)
stopindex = fs.find_closest(time, stim_dur)
validspikes = spikes[(spikes > pause) & (spikes < stim_dur)]

# compute envelope

# rectification
lower_eod_rect = eod.clip(min=0)

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

# fix noisy peaks
peaks = np.array(peaks)[beat[peaks] < -0.6]

# interpolate between peaks
interp = interpolate.interp1d(
    time[peaks], beat[peaks], kind="cubic", fill_value="extrapolate"
)
envelope = interp(time)

# extract envelope periods
env = envelope[startindex:stopindex]
env = fs.coustomnorm(env)
env_time = time[startindex:stopindex]
env_roll = np.roll(env, 1)
p_indices = np.arange(len(env_time))[(env > 0) & (env_roll <= 0)]

# collect times
envs_ts = []
for i, ii in zip(p_indices, p_indices[1:]):

    tmin, tmax = env_time[i], env_time[ii]  # make bounds to select spikes in
    t = env_time[i:ii] - tmin  # collect envelope time
    envs_ts.append(t)  # add to lists

# get max and min duration
abs_tmin = np.min(fs.flatten(envs_ts))
abs_tmax = np.max(fs.flatten(envs_ts))

# make time array
envs_ts = np.arange(abs_tmin, abs_tmax, dt)

# collect envelopes and spikes
envs = []
spks = []
for i, ii in zip(p_indices, p_indices[1:]):

    # get data
    e = env[i:ii]  # collect envelope
    tmin, tmax = env_time[i], env_time[ii]  # make bounds to select spikes in
    s = spikes[(spikes > tmin) & (spikes < tmax)]  # get spikes
    s = s - tmin  # norm spikes to start at 0

    # nanpad envelope
    if len(e) != len(envs_ts):
        dright = fs.find_closest(envs_ts, tmax - tmin)
        dnan = len(envs_ts) - dright
        nans = np.full(dnan, np.nan)
        e = np.append(e, nans)

    # add to lists
    envs.append(e)
    spks.append(s)

meanbeat = np.nanmean(np.array(envs), axis=0)

# compute gamma kde with coustom time array
kdetime = np.linspace(
    abs_tmin,
    abs_tmax,
    1000,
)

# compute kde
kdes = [fs.acausal_kde1d(s, kdetime, width=0.002) for i, s in enumerate(spks)]
meankde = np.mean(kdes, axis=0)

jit_x1 = np.ones_like(fs.flatten(spks)) * -4

# plot
fig, ax = plt.subplots(2, 1, figsize=(16 * ps.cm, 12 * ps.cm), sharex=True)
for e in envs:
    ax[0].plot(envs_ts * 1000, e, c="darkgrey", alpha=0.1)
ax[0].plot(envs_ts * 1000, meanbeat, color="k")
ax[0].axis("off")

ax[1].scatter(
    np.array(fs.flatten(spks)) * 1000, jit_x1, alpha=1, zorder=10, c="k", marker="|"
)
ax[1].fill_between(
    kdetime * 1000, np.zeros_like(meankde), meankde, color="darkgrey", alpha=1
)

ax[1].set_xlabel("Time [ms]")
ax[1].set_ylabel("Rate [Hz]")

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

ax[1].set_xticks(range(0, 30, 5))
ax[1].set_yticks(range(0, 35, 10))

ax[1].spines.left.set_bounds((0, 30))
ax[1].spines.bottom.set_bounds((0, 25))
plt.show()
