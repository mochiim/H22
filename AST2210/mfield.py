import numpy as np
import matplotlib.pyplot as plt

def epotlist(r,Q,R):
    """
    Function to calculate electric potential in an arbitrary point r given by lists of Q and R
    r = end point
    R = starting point (list)
    Q = charges (list)
    1/4*pi*epsilon_0 = c = 1, meaning calculations are done in units of c
    """
    V = 0 # electric potential "storage" for calculated values
    for i in range(len(R)):
        Ri = r - R[i] # distance between charge and observation point
        Qi = Q[i] # electric charge in point i
        Rinorm = np.linalg.norm(Ri) # length of vector Ri
        if Rinorm > 0.0046547454:
            V += Qi/Rinorm
        else:
            V += 0
    return V

# create a new meshgrid
L = 0.006
N = 50
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
xx, yy = np.meshgrid(x, y)

# Setting up list of charges in our system

R = [] # list for positions
Q = [] # list of charges
r1 = np.array([-0.000001, 0]); q1 = 1000
r2 = np.array([0.000001, 0]); q2 = -1000
R.append(r1); Q.append(q1)
R.append(r2); Q.append(q2)

# Calculate the field at each grid point, N
V = np.zeros((N, N),float)

for i in range(len(xx.flat)): # running through length of xx when flat
    r = np.array([xx.flat[i], yy.flat[i]]) # a point in our grid
    V.flat[i] = epotlist(r, Q, R) # calculating the eletric potential

from time import sleep
from tqdm import tqdm
for i in tqdm(range(10)):
    sleep(3)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
magnetic = ax.contourf(xx, yy, V, 200, cmap = "hot")
ax.set_aspect("equal")
fig.colorbar(magnetic)
plt.show()
