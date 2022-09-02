import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sc

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

slope_t, intercept_t, r_t, p_t, se_t = sc.stats.linregress(time, T_t)
slope_b, intercept_b, r_b, p_b, se_b = sc.stats.linregress(time, T_b)


fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(time, T_t, label = "Mug 1")
ax1.plot(time, slope_t*time + intercept_t)
ax1.set_ylabel("Temperature [c]")



ax2.plot(time, T_b, label = "Mug 2")
ax2.plot(time, slope_b*time + intercept_b)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Temperature [c]")
plt.legend()
#plt.savefig("temperature1.png")
plt.show()



#print(os.getcwd())
