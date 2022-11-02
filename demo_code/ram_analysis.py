import random
from turtle import st

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

import functions as fs
from termcolors import TermColor as tc

def spike_triggered_average(spikes, stimulus, dt, t_min=-0.1, t_max=0.1):
    """ Estimate the spike-triggered-average (STA) stimulus.
    Parameters
    ----------
    spikes: ndarray of floats
    Spike times of a single trial. stimulus: ndarray of floats
    The stimulus. 
    dt: float
    Temporal resolution of the stimulus. 
    t_min: float
    The time before the spike that should be taken into account.
    Same unit as `dt`. 
    t_max: float
    The time after the spike that should be taken into account. Same unit as `dt`.
    Returns
    -------
    time: ndarray of floats
    Time axis of the STA. sta: ndarray of floats
    Spike-triggered-average. sd: ndarray of floats
    Corresponding standard deviation. count: int
    Number of spikes used to computed the STA. """
    count = 0
    time = np.arange(t_min, t_max, dt)  # time for the STA                              
    snippets = np.zeros((len(time), len(spikes))) 
    for t in spikes:                                                    # for each spike
        min_index = int(np.round((t+t_min)/dt))                             # start index of snippet # end index of snippet
        max_index = min_index + len(time)                                       # snippet not fully contained in stimulus, skip it:
        if (min_index < 0) or (max_index > len(stimulus)):
            continue

        snippets[:,count] = stimulus[min_index:max_index] # store snippet
        count += 1

    sta = np.mean(snippets[:,:count], axis=1) # average and
    sd = np.std(snippets[:,:count], axis=1) # standard deviation over all snippets
    return time, sta, sd, count


# plot 15 seconds of Baseline activity 
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
ram = d['FileStimulus_3']
stimulie_rlx = ram.stimuli
ram.stimulus_folder = "../data/stimulus/"
s, t = ram.load_stimulus()

stas = []
sds = []
spike_times = []
for stim in stimulie_rlx:
    spikes = stim.trace_data('Spikes-1')[0]
    dt = t[1]- t[0]
    time, sta, sd, count = spike_triggered_average(spikes, s, dt, t_min=-0.025, t_max=0.025)
    stas.append(sta)
    spike_times.append(spikes)
    sds.append(sd)
    
mean_stas = np.array(np.mean(stas, axis=0))
mean_sds = np.array(np.std(sds, axis=0))

fig, ax = plt.subplots()
ax.plot(time, mean_stas)
ax.fill_between(time, mean_stas-mean_sds, mean_stas+mean_sds, alpha=0.3, zorder=-10)
ax.hlines(0, -10, 10, linestyles='dashed', alpha=0.6, color="k")
ax.vlines(0, -10, 10, linestyles='dashed', alpha=0.6, color="k")
ax.set_xlim(-0.03, 0.03)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Stimulus")
plt.show()

