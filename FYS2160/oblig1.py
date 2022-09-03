import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sc
import random

time = [] # [s]
T_t = [] # Temperature of water inside Temperfect mug
T_b = [] # Temperature of water inside bodum thermos mug

with open('termokopper.txt', 'r') as lines:
    for line in lines:
        row = line.split()
        time.append(float(row[0]))
        T_t.append(float(row[1]))
        T_b.append(float(row[2]))

time = np.array(time)
T_t = np.array(T_t)
T_b = np.array(T_b)

slope_t, intercept_t, r_t, p_t, se_t = sc.stats.linregress(time, T_t) # tau Temperfect
slope_b, intercept_b, r_b, p_b, se_b = sc.stats.linregress(time, T_b) # tau bodum

Ctb = 5
N = 80000
nstep = 15*N
tau = N
Tt = np.zeros(nstep).transpose()
Tb = np.zeros(nstep).transpose()
Tt[0] = 1
Tb[0] = -1
Tr = -1

for i in range(1, nstep + 1):
    r = 4*random.randint(1, 1) - 2
    DT = Tt[i - 1] - Tb[i - 1]
    if (r < DT):
        Tt[i] = Tt[i - 1] - 1/N
        Tb[i] = Tb[i - 1] + Ctb/N
    else:
        Tt[i] = Tt[i-1] + 1/N
        Tb[i] = Tb[i-1] - Ctb/N


# Temperature plot over time of both mugs
"""
plt.plot(time, T_t, label = "Mug 1")
plt.plot(time, T_b, label = "Mug 2")
plt.xlabel("Time [s]", fontsize = 18)
plt.ylabel("Temperature [°C]", fontsize = 18)
plt.legend(prop={'size': 12})
plt.savefig("temp.png")
plt.show()
"""

"""
plt.plot(time, T_t, label = "Mug 1")
plt.plot(time, slope_t*time + intercept_t, label = rf"Linear regression, $\tau$ = {slope_t}")
plt.xlabel("Time [s]", fontsize = 18)
plt.ylabel("Temperature [°C]", fontsize = 18)
plt.legend(prop={'size': 12})
plt.savefig("tau_t.png")
plt.show()
"""

"""
plt.plot(time, T_b, label = "Mug 2")
#plt.plot(time, slope_b*time + intercept_b, label = rf"Linear regression, $\tau$ = {slope_b}")
plt.xlabel("Time [s]", fontsize = 18)
plt.ylabel("Temperature [°C]", fontsize = 18)
plt.legend(prop={'size': 12})
#plt.savefig("tau_b.png")
plt.show()
"""

#print(os.getcwd())
