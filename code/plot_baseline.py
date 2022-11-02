
import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
from plotstyle import PlotStyle

import functions as fs
from termcolors import TermColor as tc

ps = PlotStyle()

# plot 15 seconds of Baseline activity 
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")

fig, ax = plt.subplots()
fs.plot_baseline(ax, d, end=15.0)
fs.doublesave("../figures/baseline_acitivity")

plt.show()



