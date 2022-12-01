import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

datafile = "/Users/rebeccanguyen/Documents/GitHub/H22/channel_current.txt"

# Channels containing currents with unit pA (10^{−12}A)
chan1 = []
chan2 = []
chan3 = []
chan4 = []
chan5 = []
chan6 = []
chan7 = []
chan8 = []
chan9 = []

tau = [0.20, 0.45, 0.70, 0.95, 1.20, 1.45, 1.70, 1.95, 2.20] # tensions [pN/nm]

with open (datafile) as infile:
    for line in infile:
        separate = line.split()
        chan1.append(eval(separate[0]))
        chan2.append(eval(separate[1]))
        chan3.append(eval(separate[2]))
        chan4.append(eval(separate[3]))
        chan5.append(eval(separate[4]))
        chan6.append(eval(separate[5]))
        chan7.append(eval(separate[6]))
        chan8.append(eval(separate[7]))
        chan9.append(eval(separate[8]))

def plothist(channel, taunum):
    plt.hist(channel, 100)
    plt.title(f"tau = {tau[taunum]}")
    plt.xlabel("Current, pA")
    plt.ylabel("Counts")
    #plt.savefig(f"tau{tau[taunum]}.png")


# Experimental open probability, Po, at each tension
# by choosing a current threshold between open and closed states

"""
plothist(chan1, 0)
plothist(chan2, 1)
plothist(chan3, 2)
plothist(chan4, 3)
plothist(chan5, 4)
plothist(chan6, 5)
plothist(chan7, 6)
plothist(chan8, 7)
plothist(chan9, 8)
"""
deltae = -5.0 *sc.k*1-1e-9*1e-12 # [kT] -> [PN*nm]
deltae_error = 1.1 *sc.k*1-1e-9*1e-12 # [kT] -> [PN*nm]
deltaA = 10 # [nm^2]
threshold = -1.75 # current threshold

def P0(tauval):
    tau = tauval # tension [pN/nm]
    P_0 = 1/(np.exp(deltae - tau*deltaA) + 1)
    return P_0

def P0error(tauval): # +
    tau = tauval # tension [pN/nm]
    P_0error = 1/(np.exp((deltae+deltae_error) - tau*deltaA) + 1)
    return P_0error

def P0errorminus(tauval): # -
    tau = tauval # tension [pN/nm]
    P_0error = 1/(np.exp((deltae-deltae_error) - tau*deltaA) + 1)
    return P_0error

polist = []

for i in range(9):
    polist.append(P0(tau[i]))

polisterror = []
for i in range(9):
    polisterror.append(P0error(tau[i]))

polisterrorminus = []
for i in range(9):
    polisterrorminus.append(P0errorminus(tau[i]))

def counter(channel):
    count = 0
    for i in range(len(channel)):
        if channel[i] < threshold:
            count += 1
    return count

P0list = []

P0list.append(counter(chan1)/len(chan1))
P0list.append(counter(chan2)/len(chan2))
P0list.append(counter(chan3)/len(chan3))
P0list.append(counter(chan4)/len(chan4))
P0list.append(counter(chan5)/len(chan5))
P0list.append(counter(chan6)/len(chan6))
P0list.append(counter(chan7)/len(chan7))
P0list.append(counter(chan8)/len(chan8))
P0list.append(counter(chan9)/len(chan9))


plt.plot(tau, P0list, label = "Experimental")
plt.plot(tau, polist, label = "Model")
#plt.fill_between(tau, np.array(P0list) + deltae_error, np.array(P0list) - deltae_error, alpha=0.2)
#plt.fill_between(tau, polisterror, polisterrorminus, alpha=0.2)
plt.xlabel(r"$\tau$")
plt.ylabel("Probability")
plt.title("Probability of a mechanosensitive channels being open")
plt.legend()
#plt.savefig("datavsexp.png")
plt.savefig("somedata.png")
plt.show()
