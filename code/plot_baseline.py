import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed

import functions as fs

from plotstyle import PlotStyle
from termcolors import TermColor as tc

ps = PlotStyle()

ps = PlotStyle()

# plot 15 seconds of Baseline activity
d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
figsize_x = 20 * ps.cm
figsize_y = 12 * ps.cm

fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
fs.plot_baseline(ax, d, end=10.0)
fs.doublesave("../figures/baseline_acitivity")

plt.show()

# plot Baseline with Bursts

fig2, ax2 = plt.subplots(figsize=(figsize_x, figsize_y))
fs.plot_baseline(ax2, d, end=10.0, burst=True, single=True)
fs.doublesave("../figures/burst_activity")

fig3, ax3 = plt.subplots(figsize=(figsize_x, figsize_y))
fs.plot_baseline(ax3, d, end=1, burst=True, single=True)
fs.doublesave("../figures/burst_activity_close")
