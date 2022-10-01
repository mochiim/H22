import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import os

# spectral data observed as a function of wavelength for a rectangular region
idata = np.load("/Users/rebeccanguyen/Documents/GitHub/H22/idata_square.npy")

# wavelength value for each of the 8 locations
spect_pos = np.load("/Users/rebeccanguyen/Documents/GitHub/H22/spect_pos.npy")

wav_idx = 4 #or any other index in the range, here between 0 and 7
intensity_data = idata[:,:,wav_idx]

def wavspec(nested_coor):
    """
    Wavelength spectrum for a given point (x, y)

    nested_coor contains (x, y) coordiantes at some location
    """
    x = nested_coor[0]
    y = nested_coor[1]
    wavelength_spectrum = idata[y-1,x-1,:]
    return wavelength_spectrum
# We will look at the spectra at some specific pixel locations
A = (49, 197)
B = (238, 443)
C = (397, 213)
D = (466, 52)

# sub field of view
h = 100 # px
w = 150 # px
x1 = 525 # x pixel of lower left corner
y1 = 325 # y pixel of lower left corner

# slice out small grid
corner = (x1-1, y1-1)
height = h
width = w
rect = Rectangle(corner, width, height, linewidth = 2, edgecolor = "yellow", facecolor = "none")
x1, y1 = rect.get_xy()
x2 = x1 + rect.get_width()
y2 = y1 + rect.get_height()
slice_x = slice(x1,x2)
slice_y = slice(y1,y2)
idata_cut = idata[slice_y, slice_x, :]

# intensity plot
plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "hot"

fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(intensity_data)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Intensity", fontsize = 18)
ax.set_title("Intensity", fontsize = 18)
ax.set_xlabel("x [idx]", fontsize = 18)
ax.set_ylabel("y [idy]", fontsize = 18)
ax.add_patch(rect) # sub field of view
fig.tight_layout()
#plt.savefig("intensitysub.png")
plt.show()

def specline(wavelength_spectrum, name):
    """
    Creating plot of spectral line from a wavelength spectrum
    """
    fig, ax = plt.subplots()
    ax.plot(wavelength_spectrum, ls = "--", lw = 1, color = "red", marker = "x", label = "intensity observed as a function of wavelength")
    ax.set_title(f"Spectra for point {name}", fontsize = 18)
    ax.set_xlabel(r"Wavelength $\lambda_i$", fontsize = 18)
    ax.set_ylabel("Intensity", fontsize = 18)
    ax.legend(prop={'size': 12})
    fig.tight_layout()
    #plt.savefig(f"spectra{name}.png")
    plt.show()

specline(wavspec(A), "A")
specline(wavspec(B), "B")
specline(wavspec(C), "C")
specline(wavspec(D), "D")
