import random

import matplotlib.pyplot as plt
import numpy as np
import rlxnix as rlx
from IPython import embed
import plot_parameter

import functions as fs
from termcolors import TermColor as tc


d = rlx.Dataset("../data/2022-10-27-ai-invivo-1.nix")

d.plot_timeline()

d = rlx.Dataset("../data/2022-10-27-ah-invivo-1.nix")

d.plot_timeline()