import numpy as np
import matplotlib.pyplot as plt

v = [] # velocity [km/s]
Ta = [] # antenna temperature

datafile = "/Users/rebeccanguyen/Documents/GitHub/H22/IRAS13120_spec.txt"

with open (datafile) as infile:
    for line in infile:
        separate = line.split() # separate elements
        v.append(eval(separate[0]))
        Ta.append(eval(separate[1]))

def KtoJy(Ta):
    """
    Conversion from antenna temperature to flux density, T_A[K] → Sν[Jy],
    """
    eta_a = 0 # aperture efficiency
    Gaminv = 24.4152/eta_a # conversion factor reported on the APEX website [Jy/K^-1]
    return Ta*Gaminv

plt.plot(v, Ta)
plt.show()
