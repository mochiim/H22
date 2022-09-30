import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import os

idata = np.load("idata_square.npy")
spect_pos = np.load("spect_pos.npy")
