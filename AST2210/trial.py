import numpy as np
from scipy.integrate import simpson
from numpy import trapz
import matplotlib.pyplot as plt

# The y values.  A numpy array is used here,
# but a python list could also be used.
v = [] # velocity [km/s]
Ta = [] # antenna temperature

Gaminv = 36 # conversion factor from antenna temperature to flux density

datafile = "/Users/rebeccanguyen/Documents/GitHub/H22/IRAS13120_spec.txt"

with open (datafile) as infile:
    for line in infile:
        separate = line.split() # separate elements
        v.append(eval(separate[0]))
        Ta.append(eval(separate[1]))

v = np.array(v)
y = np.array(Ta)*Gaminv
# Compute the area using the composite trapezoidal rule.
area = trapz(y, dx=5)
print("area =", area)

# Compute the area using the composite Simpson's rule.
area = simpson(y, dx=5)
print("area =", area)

from scipy import integrate
f = lambda x: x**8
a = integrate.quadrature(f, 0.0, 1.0)
print(a)
