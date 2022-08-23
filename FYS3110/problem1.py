import numpy as np
import matplotlib.pyplot as plt

epsilon = np.array([0.01, 0.1, 1])
x = np.linspace(-1, 1, 100)

def dirac_delta(epsilon, x):
    return (1/np.pi)*(epsilon/(epsilon**2 + x**2))

figure, ax = plt.subplots(1, 1)
for i in range(3):
    plt.plot(dirac_delta(epsilon[i], x), label = rf"$\epsilon$ = {epsilon[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
