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
    wavelength_spectrum = idata[y - 1, x - 1,:]
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
corner = (x1 - 1, y1 - 1)
height = h
width = w
rect = Rectangle(corner, width, height, linewidth = 2, edgecolor = "yellow", facecolor = "none")
x1, y1 = rect.get_xy()
x2 = x1 + rect.get_width()
y2 = y1 + rect.get_height()
slice_x = slice(x1,x2)
slice_y = slice(y1,y2)
idata_cut = idata[slice_y, slice_x, wav_idx]

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
    #plt.savefig("intensitypoints.png")

# intensity plot
"""
inplot()
plt.scatter(A[0], A[1], label  = "A", color = "dodgerblue")
plt.scatter(B[0], B[1], label  = "B", color = "azure")
plt.scatter(C[0], C[1], label  = "C", color = "violet")
plt.scatter(D[0], D[1], label  = "D", color = "midnightblue")
plt.legend()
"""

# sub field of view
"""
fig, ax = plt.subplots()
ax.grid(False)
im = ax.imshow(idata_cut)
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"Intensity", fontsize = 18)
ax.set_title("Intensity in sub field of view", fontsize = 18)
ax.set_xlabel("x [idx]", fontsize = 18)
ax.set_ylabel("y [idy]", fontsize = 18)
ax.add_patch(rect) # sub field of view
fig.tight_layout()
plt.show()
"""

avg = np.mean(idata, axis = (0, 1)) # average spectra

def gaussian(x, d, a, b, c):
    """
    Gaussian fitting of emission and absorption lines
    a: the amplitude of the gaussian.
    b: the mean of the gaussian. The position of the peak on the x-axis.
    c: the standard deviation.
    d: the constant term, y-value of the baseline.
    """
    return d + a * np.exp(-(x - b)**2 / (2 * c**2))

def fitting(y):
    """
    y: spectra line to be fitted with a gaussian curve
    """
    x = np.linspace(0, 7, 8)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = curve_fit(gaussian, x, y, p0 = [min(y), max(y), sigma, 2])
    x = np.linspace(0, 7, 1000)
    return x, popt

# plotting spectral lines and their gaussian fitting of 4 points in a sub plot
"""
fig, ax = plt.subplots(2, 2)
x = np.linspace(0, 7, 1000)

poptA = fitting(wavspec(A))
ax[0, 0].plot(wavspec(A), ls = "--", lw = 1, color = "red", marker = "x", label = "Data")
ax[0, 0].plot(x, gaussian(x, *poptA), color = "blue", label = "Gaussian fitting")
ax[0, 0].set_title(f"Fitting for spectra A", fontsize = 20)
ax[0, 0].set_xlabel(r"Wavelength $\lambda_i$", fontsize = 20)
ax[0, 0].set_ylabel("Intensity", fontsize = 20)
ax[0, 0].legend(prop={'size': 12})

poptB = fitting(wavspec(B))
ax[0, 1].plot(wavspec(B), ls = "--", lw = 1, color = "red", marker = "x", label = "Data")
ax[0, 1].plot(x, gaussian(x, *poptB), color = "blue", label = "Gaussian fitting")
ax[0, 1].set_title(f"Fitting for spectra B", fontsize = 20)
ax[0, 1].set_xlabel(r"Wavelength $\lambda_i$", fontsize = 20)
ax[0, 1].set_ylabel("Intensity", fontsize = 20)
ax[0, 1].legend(prop={'size': 12})

poptC = fitting(wavspec(C))
ax[1, 0].plot(wavspec(C), ls = "--", lw = 1, color = "red", marker = "x", label = "Data")
ax[1, 0].plot(x, gaussian(x, *poptC), color = "blue", label = "Gaussian fitting")
ax[1, 0].set_title(f"Fitting for spectra C", fontsize = 20)
ax[1, 0].set_xlabel(r"Wavelength $\lambda_i$", fontsize = 20)
ax[1, 0].set_ylabel("Intensity", fontsize = 20)
ax[1, 0].legend(prop={'size': 12})

poptD = fitting(wavspec(D))
ax[1, 1].plot(wavspec(D), ls = "--", lw = 1, color = "red", marker = "x", label = "Data")
ax[1, 1].plot(x, gaussian(x, *poptD), color = "blue", label = "Gaussian fitting")
ax[1, 1].set_title(f"Fitting for spectra D", fontsize = 20)
ax[1, 1].set_xlabel(r"Wavelength $\lambda_i$", fontsize = 20)
ax[1, 1].set_ylabel("Intensity", fontsize = 20)
ax[1, 1].legend(prop={'size': 12})

fig.tight_layout()
plt.show()


"""
x, popt = fitting(avg)
plt.plot(x, gaussian(x, *popt), color = "blue", label = "Gaussian fitting")
plt.plot(avg, ls = "--", lw = 1, color = "red", marker = "x")
plt.show()
