import numpy as np 
import matplotlib.pyplot as plt 
import rlxnix as rlx 
from IPython import embed
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from functions import sort_reodfs

dset = rlx.Dataset("../data/2021-11-11-af-invivo-1.nix")

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

n = 3 
zeros = np.where(np.diff(np.sign(stim)))[0]
maxima_index = []
for i, x in enumerate(range(len(zeros))):
    #maxima = np.argmax(np.arange(stim[zeros[i]], stim[zeros[i+2]], dt))
     if i % n == 0:
        max_range = np.arange(stim_time[zeros[x]], stim_time[zeros[x+2]],dt)
        max_index = range(len(max_range))
        maximum = np.argmax(stim[max_index])
        maxima_index.append(maximum)



embed()
exit()

plt.plot(stim_time[t_times], stim[t_times], alpha=0.6)
for i in range(multiple_ts):
    plt.scatter(t*i, np.zeros(1)*0.09)
plt.show()

