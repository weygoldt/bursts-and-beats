import numpy as np 
import matplotlib.pyplot as plt 
import rlxnix as rlx 
from IPython import embed

dset = rlx.Dataset("data/2021-11-11-af-invivo-1.nix")

chirp1 = dset["Chirps_8"]
frq_dset = dset.metadata["Recording"]["Subject"]["EOD Frequency"][0][0]
reodf = chirp1.relative_eodf
# beat frequency 
freq_stim = frq_dset * reodf
# Window T with freq_stim
t = 1/freq_stim

chirp1_stim_count = chirp1.stimulus_count

stim, stim_time = chirp1.stimulus_output(0)
multiple_ts = 1
dt = stim_time[1]-stim_time[0]
t_time = np.arange(0, t*multiple_ts, dt)
t_times = range(len(t_time))

plt.plot(stim_time[t_times], stim[t_times], alpha=0.6)
for i in range(multiple_ts):
    plt.scatter(t*i, np.zeros(1)*0.09)
plt.show()


