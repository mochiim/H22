import numpy as np
import matplotlib.pyplot as plt

time = [] # [s]
T_b = [] # Temperature of water inside bodum thermos mug
T_t = [] # Temperature of water inside Temperfect mug

with open("termokopper.txt") as lines:
    for line in lines:
        row = line.split()
        time.append(row[0])
        T_b.append(row[1])
        T_t.append(row[2])