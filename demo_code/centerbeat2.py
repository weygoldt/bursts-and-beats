import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx

import functions as fs
from termcolors import TermColor as tc

# get data
data = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
sams = [i for i in data.repros if "SAM" in i]
sams = sams[2:]
sam = sams[2]
samd = data[sam]
spikes = samd.trace_data("Spikes-1")[0]
beat, time = samd.trace_data("LocalEOD-1")
pause = 5
stim_dur = 30

# start at 5 seconds and cut next 30 seconds into period bins
