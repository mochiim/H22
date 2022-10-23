import numpy as np
import matplotlib.pyplot as pyplot

arckm = 740 # 1 arcsec = 740 km in the solar surface
arc = 0.058
x = 750 # [px]
y = 550 # [px]

"""Spatial resolution of this data is 0.058 arcsec per pixel"""
"""1 arcsec = 740 km on the solar surface (photosphere)"""

tot = (y*arc*arckm)*(x*arc*arckm)
print(tot)
