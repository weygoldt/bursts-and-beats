import numpy as np 
import rlxnix as rlx 
import matplotlib.pyplot as plt
from IPython import embed


def filter_stimulus(stim):
    contrast = stim.feature_data("rectangle-1_Contrast")
    return contrast > 9.5 and contrast < 12.9

data = rlx.Dataset("data/2022-10-27-aa-invivo-1.nix")
#fi = data.find_stimuli("FIC", filter_stimulus)
fi = data['FICurve_1']
stim_count = fi.stimulus_count

contrasts = []
for count in range(stim_count):
    contrast = fi[count].feature_data("rectangle-1_Contrast")[0]
    contrasts.append(contrast)

v, t = fi[0].trace_data('LocalEOD-1', before=0.2, after=0.2)
spike, _ = fi[3].trace_data('Spikes-1')

peaks = []
for s in range(stim_count):
    spike, _ = fi[s].trace_data('Spikes-1')
    peaks.append(spike)

    spikes = []
    for p in peaks:
        if p <= 0.1:
            spike.append(p)

    
contrast = fi[2].feature_data("rectangle-1_Contrast")

embed()
exit()



