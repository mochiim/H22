import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sc
import random
from math import factorial
import scipy.constants as scc

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

################

slope_t, intercept_t, r_t, p_t, se_t = sc.stats.linregress(time, T_t) # tau Temperfect
slope_b, intercept_b, r_b, p_b, se_b = sc.stats.linregress(time, T_b) # tau bodum

################

k = 1 #scc.Boltzmann
N = int(300) # a suitable numbers of oscillators
nsteps = 800
q = np.linspace(1, nsteps, nsteps) # number of energy units
dq = (q[-1]-q[0])/(len(q)-1)

omega = np.zeros(nsteps) # multiplicity of an Einstein solid
for i in range(nsteps):
    omega[i] = factorial(int(q[i]) + N - 1)/(factorial(int(q[i]))*factorial(N - 1))

S = k*np.log(omega) # entropy

dqdS = np.zeros(nsteps)
for i in range(1, nsteps):
    dqdS[i] = dq/(S[i] - S[i - 1])

T = dqdS # temperature

Cv = np.zeros(nsteps) # heat capacity
for i in range(1, nsteps):
    Cv[i] = dq/(T[i] - T[i -1])

Ctb = 5
N = 80000
nstep = 15*N
tau = N
Tt = np.zeros(nstep)
Tb = np.zeros(nstep)
Tt[0] = 1
Tb[0] = -1
Tr = -1

for i in range(1, nstep):
    r = 4*random.uniform(1, 1) - 2
    DT = Tt[i - 1] - Tb[i - 1]
    if (r < DT):
        Tt[i] = Tt[i - 1] - 1/N
        Tb[i] = Tb[i - 1] + Ctb/N
    else:
        Tt[i] = Tt[i-1] + 1/N
        Tb[i] = Tb[i-1] - Ctb/N

plt.plot(np.linspace(0, nstep, nstep)/tau, Tt, "r")
plt.plot(np.linspace(0, nstep, nstep)/tau, Tb, "k")
plt.show()

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

# Plot tau of Temperfect
"""
plt.plot(time, T_t, label = "Mug 1")
plt.plot(time, slope_t*time + intercept_t, label = rf"Linear regression, $\tau$ = {slope_t}")
plt.xlabel("Time [s]", fontsize = 18)
plt.ylabel("Temperature [°C]", fontsize = 18)
plt.legend(prop={'size': 12})
#plt.savefig("tau_t.png")
plt.show()
"""

# Plot tau of Bodun
"""
plt.plot(time, T_b, label = "Mug 2")
plt.plot(time, slope_b*time + intercept_b, label = rf"Linear regression, $\tau$ = {slope_b}")
plt.xlabel("Time [s]", fontsize = 18)
plt.ylabel("Temperature [°C]", fontsize = 18)
plt.legend(prop={'size': 12})
plt.savefig("tau_b.png")
plt.show()
"""

# plot T against S
"""
plt.plot(T, S)
plt.xlabel("Temperature [°C]", fontsize = 18)
plt.ylabel("Entropy [J/K]", fontsize = 18)
plt.savefig("T_mot_S.png")
plt.show()
"""

# plot T against Cv
"""
plt.plot(T, Cv)
plt.xlabel("Temperature [°C]", fontsize = 18)
plt.ylabel("Heat capacity [J/K]", fontsize = 18)
plt.savefig("T_mot_cv.png")
plt.show()
"""

#print(os.getcwd())
