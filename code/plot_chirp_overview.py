import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

import functions as fs
from termcolors import TermColor as tc
from plotstyle import PlotStyle
ps = PlotStyle()

d = rlx.Dataset("../data/2022-10-27-aa-invivo-1.nix")
embed()
exit()
chirp = d["Chirps_8"]
chirp.plot_overview()