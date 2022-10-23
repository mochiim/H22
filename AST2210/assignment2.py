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

print(v)
