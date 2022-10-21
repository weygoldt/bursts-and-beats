import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

# load relacs dataset
d = rlx.Dataset("../data/2022-10-20-aa-invivo-1.nix")

# print some information
type(d)
d.name
d.metadata

# plot timeline
d.plot_timeline()

# print the repros used
d.repros

# look at which data traces where recorded
d.data_traces

# look at what events where recorded
d.event_traces

# look at recording date
d.recording_date

# look at baseline repro
baseline = d["BaselineActivity_4"]
baseline.baseline_rate  # firing rate
baseline.baseline_cv  # coefficient of variation of the ISI

# look at the spikes in the baseline recording
baseline.spikes
spike_times = baseline.spikes()
plt.eventplot(spike_times)
plt.show()

# look at the membrane voltage
v, t = baseline.membrane_voltage()
plt.plot(t, v, color="black")
plt.scatter(spike_times, np.ones_like(spike_times) * (-8), marker="|", color="black")
plt.show()

# One thing to look at at baseline: How many are isolated, how many come in bursts?

# lets look at chirps
chirp_repros = d.repro_runs("chirps")
chirps = chirp_repros[0]
chirps.beat_specification
chirps.delta_f
chirps.relative_eodf
chirps.chirp_times
chirps.stimulus_count
chirps.plot_overview(0)
chirps.plot_overview(2)

# lets look at the stimulus
stim, stim_t = chirps.stimulus_output(0)
plt.plot(stim_t, stim)
plt.show()

# spike times relative to repro onset
chirps.spikes()  # all spikes

# these spikes are now for a single stimulus onset
chirps.spikes(1)  # spikes for a single stimulus
# thats why the spikes of the second print are not found in the first print

filestim = d["FileStimulus_1"]
type(filestim)
filestim.contrast

filestim.stimulus_folder = "../stimuli"
stim, stim_time = filestim.load_stimulus()
plt.plot(stim_time, stim)
plt.show()

# correlate white noise with spike times
spike_times = filestim.spikes(0)
plt.plot(stim_time, stim)
plt.scatter(spike_times, np.ones_like(spike_times) * np.max(stim))
plt.show()

for i in range(filestim.stimulus_count):
    print(filestim[i].start_time, filestim[i].duration)

# show filestim start and stop time
filestim.start_time
filestim.stop_time
d.plot_timeline()
