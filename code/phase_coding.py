import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
import pandas as pd 
from scipy import interpolate
from IPython import embed
from tqdm import tqdm

import functions as fs
from plotstyle import PlotStyle


d1 = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

chirp_repros = [i for i in d1.repros if "Chirps" in i]
before_t = 0.15
after_t = 0.15


chirp_snippets = []
repro_number = []
stimulus_number = []
chirp_number= []
for r, repro in tqdm(enumerate(chirp_repros)):
    # get chirps from each repro
    chirps = d1[repro]
    # go through every single repos chirp repition 
    for sc, i  in enumerate(range(chirps.stimulus_count)):
        # get data
        _, time = chirps.membrane_voltage(i)
        spikes = chirps.spikes(i)
        chirp_times = chirps.chirp_times[0][i]
        vlocal, t  = chirps.local_eod(i)
        dt = t[1]- t[0]
        before_indices = np.round(before_t / dt)
        after_indices = np.round(after_t / dt)
        for j, c in enumerate(chirp_times):
            c_index = fs.find_closest(t, c)
            indices = np.arange(c_index - before_indices, c_index + after_indices, dtype=int)
            chirp_snip = vlocal[indices]
            chirp_snippets.extend([chirp_snip])
            repro_number.append(r)
            stimulus_number.append(sc)
            chirp_number.append(j)

df = pd.DataFrame({"repro":repro_number,"stimulus":stimulus_number, "chirp": chirp_number, "data":chirp_snippets})


embed()
exit()
