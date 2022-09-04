import numpy as np
import matplotlib.pyplot as plt
from math import factorial
"""
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
"""
k = 1
N = int(300)
N_len = 800
q = np.linspace(1,N_len,N_len)
dq = (q[-1]-q[0])/(len(q)-1)


omega = np.zeros(N_len)
for i in range(0,N_len):
    omega[i] = factorial(int(q[i])+N-1)/(factorial(int(q[i]))*factorial(N-1))
S = k*np.log(omega)

dqdS = np.zeros(N_len)

for i in range(1,N_len):
    dqdS[i] = dq/(S[i]-S[i-1])
T = dqdS

Cv = np.zeros(N_len)
for i in range(1,N_len):
    Cv[i] = dq/(T[i]-T[i-1])


plt.plot(q,S,label=r"$N=%g$"%(N_len))
plt.xlabel("q",FontSize=15)
plt.ylabel(r"$C_V/Nk$",FontSize=15)
plt.legend()

plt.show()
