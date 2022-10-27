import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from numpy import trapz
from scipy import integrate
from sklearn.metrics import mean_squared_error
import sklearn.metrics
import astropy.constants as const
sns.color_palette("bright")
sns.set_theme()
sns.set_style("darkgrid")

### Reading data ###

v = [] # velocity [km/s]
Ta = [] # antenna temperature

Gaminv = 36 # conversion factor from antenna temperature to flux density

# galaxy IRAS 13120-5453
DL = 139.9 # luminosity distance [Mpc]
z = 0.0308 # redshift
vobs = 23658 # observerved frequency [GHz]

datafile = "/Users/rebeccanguyen/Documents/GitHub/H22/IRAS13120_spec.txt"

with open (datafile) as infile:
    for line in infile:
        separate = line.split() # separate elements
        v.append(eval(separate[0]))
        Ta.append(eval(separate[1]))

v = np.array(v)
T = np.array(Ta)*Gaminv # flux density

def gaussian(x, a, b, c, d):
    """
    Gaussian fitting of emission and absorption lines
    a: the amplitude of the gaussian.
    b: the mean of the gaussian. The position of the peak on the x-axis.
    c: the standard deviation.
    d: the constant term, y-value of the baseline.
    """
    return a*np.exp(-((x - b)**2) / (2*c**2)) + d

def fitting(x, y):
    """
    Gaussian curve fitting
    """
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = curve_fit(gaussian, x, y, (np.max(y), mean, sigma, 0))
    return popt, pcov

# gaussian curve
popt, pcov = fitting(v, T)
perr = np.sqrt(np.diag(pcov)) # uncertainty [a, b, c, d]

# calculate fwhm
fwhm = 2*np.sqrt(2 * np.log(2))*popt[2]
hm = popt[0] / 2 # half max
fw = np.linspace(popt[1] - fwhm/2, popt[1] + fwhm/2, 2) # full width

# uncertainty of fwhm
fwhmer = 2*np.sqrt(2 * np.log(2))*(popt[2]+2.30359157)
un = np.linspace((popt[1]+2.24556722) - fwhmer/2, (popt[1]+2.24556722) + fwhmer/2, 2)

# total line flux
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
g = lambda x: a*np.exp(-((x - b)**2) / (2*c**2)) + d
areagauss = integrate.quadrature(g, -1000, 1000) # (total line flux, error)

# S/N
mse = sklearn.metrics.mean_squared_error(T, gaussian(v, *popt))
rmse = np.sqrt(mse)
SN = popt[0]/rmse
SNerror = ((popt[0] + perr[0])/rmse) - SN

def lineluminosity(vobs, z, DL, lineflux):
    """
    vobs = observed frequency
    z = redshift
    DL = luminosity distance
    lineflux = total line flux
    """
    return (3.25*1e7)*(DL**2 / (vobs**2*((1 + z)**3)))*lineflux

linelum = lineluminosity(vobs, z, DL, areagauss[0])
linelumerror = lineluminosity(vobs, z, DL, areagauss[0] + areagauss[1])
print(f"Line luminosity: {linelum} +- {linelumerror - linelum} ) [K km/s pc^2")



def totmolgasmas(lco, lcoerror):
    """
    Total molecular gas mass
    """
    aco = 1.7 # [(K km s^-1 pc^2)^-1]
    acoerror = 0.4*const.M_sun.value # solar mass in unit kg
    return aco*lco, ((aco+acoerror)*(lco+lcoerror)) - (aco*lco)

tmgm, er = totmolgasmas(linelum, linelumerror)
print(f"Total moelcular gas mas: {tmgm} +- {er} [(K km s^-1 pc^2)^-1]")



"""
#plt.plot(v, T, label = "data")
plt.plot(v, gaussian(v, *popt), label = "Gaussian fit")
plt.xlabel("Velocity [km/s]")
plt.ylabel("Flux density [Jy]")
#plt.title("Gaussian fit of data")
plt.title(r"Total line flux $\int_{\Delta \nu} = S_\nu d\nu$")
#plt.legend()
#plt.savefig("gaussian.png")
plt.fill_between(v, gaussian(v, *popt), step="pre", alpha=0.4, color = "orange", label = "Area under curve")
plt.text(1, 1.5, f"Total line flux: {areagauss[0]: .2f} $\pm$ {areagauss[1]: .2g} [Jy km/s]")
plt.legend()
# plt.savefig("lineflux.png")


plt.plot(fw, np.array([hm, hm]), color='black', ls='dashed', lw=1)
plt.text(popt[1] + 5, popt[0], '$S_\\nu^{peak}=$' + f'{popt[0]:.2f} $\pm$ ' + f'{perr[0]:.3f} Jy')
plt.text(fw[1], hm, f"$FWHM$ = {fw[1] - fw[0]: .2f} $\pm$ {abs((fw[1]-fw[0]) - (un[1]-un[0])): .3f} km/s" )
plt.title("$S_\\nu^{peak}$ and FWHM")
#plt.savefig("peakfwhm.png")

plt.plot(v, T)
plt.xlabel("Velocity [km/s]")
plt.ylabel("Flux density [Jy]")

"""

#plt.show()
