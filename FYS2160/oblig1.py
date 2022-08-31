import numpy as np
import matplotlib.pyplot as plt
import os

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

plt.plot(time, T_t, label = "Temperfect mug")
plt.plot(time, T_b, label = "Bodum mug")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [c]")
plt.legend()
plt.savefig("temperature.png")
plt.show()



print(os.getcwd())
