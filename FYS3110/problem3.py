import numpy as np
import matplotlib.pyplot as plt

g = 1
V = 5
L = 25

x = np.zeros((L, L))

for i in range(L):
    x[i][i] = i

val, vec = np.linalg.eig(x)
