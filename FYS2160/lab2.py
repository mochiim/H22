import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

T =  [76.7, 80.9, 82.2, 86.2, 88.3, 91, 92.3, 93.5, 95.6, 97.7, 99.3] # boiling temperatures
P = [43.1, 56.3, 62.8, 64, 75.1, 79.9, 80.8, 90.3, 91.1, 101.4, 101.4] # pressure

T = np.array(T) + 273.5
slope, intercept, r, p, se = stats.linregress(1/T, np.log(P))
R = 8.314

x = np.linspace(0.010, 0.0130, 1000)
plt.scatter(1/T, np.log(P), color = "red", label = "Data")
plt.plot(x, slope*x + intercept, color = "blue", label = "Linear regression")
plt.xlabel("ln(P)")
plt.ylabel("1/T")
#plt.show()

Hv = -slope*R
Tru = Hv/(R*T)
print(slope)
print(Hv)
print(Tru)
