import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import seaborn as sns
sns.color_palette("bright")
sns.set_theme()
sns.set_style("darkgrid")

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

def inplot():
    """
    intensity plot
    """
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
    #plt.savefig("intensity.png")
    plt.show()

#inplot()

# creating subplot figure
#fig, ax = plt.subplots(2, 2)
#fig.tight_layout()
def specline(wavelength_spectrum, name, i, j):
    """
    Creating plot of spectral line from a wavelength spectrum
    """
    ax[i, j].plot(wavelength_spectrum, ls = "--", lw = 1, color = "red", marker = "x", label = "intensity observed as a function of wavelength")
    ax[i, j].set_title(f"Spectra for point {name}", fontsize = 18)
    ax[i, j].set_xlabel(r"Wavelength $\lambda_i$", fontsize = 18)
    ax[i, j].set_ylabel("Intensity", fontsize = 18)
    #ax[i, j].legend(prop={'size': 12})

# plotting spectral line of 4 points in a sub plot
#specline(wavspec(A), "A", 0, 0)
#specline(wavspec(B), "B", 0, 1)
#specline(wavspec(C), "C", 1, 0)
#specline(wavspec(D), "D", 1, 1)

#plt.savefig("spectrallines.png")
#plt.show()

#plt.plot(np.mean(idata, axis = (0, 1)))
#plt.show()

def gaussian(x, d, a, b, c):
    """
    Gaussian fitting of emission and absorption lines
    a: the amplitude of the gaussian.
    b: the mean of the gaussian. The position of the peak on the x-axis.
    c: the standard deviation.
    d: the constant term, y-value of the baseline.
    """
    return d + a * np.exp(-(x - b)**2 / (2 * c**2))

x = np.linspace(0, 7, 8)
y = wavspec(C)
mean = sum(x * y) / sum(y)
sigma = sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
popt, pcov = curve_fit(gaussian, x, y)
plt.plot(x, gaussian(x, *popt), 'r-')
plt.plot(y, label = "data")


"""
popt, pcov = curve_fit(Gauss, x, y, p0=[-2000, max(y), mean, sigma])
plt.plot(y, ls = "--", lw = 1, color = "red", marker = "x")
plt.plot(x, Gauss(x, *popt), 'r-', color = "blue", label='fit')
"""
plt.legend()
plt.show()
