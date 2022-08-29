import numpy as np
import matplotlib.pyplot as plt
import os

time = [] # [s]
T_b = [] # Temperature of water inside bodum thermos mug
T_t = [] # Temperature of water inside Temperfect mug

with open('termokopper.txt', 'r') as lines:
    for line in lines:
        row = line.split()
        time.append(row[0])
        T_b.append(row[1])
        T_t.append(row[2])

time = np.array(time)
T_b = np.array(T_b)
T_t = np.array(T_t)

print(time)
#print(os.getcwd())
