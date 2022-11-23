import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Rectangle

filename = "/Users/rebeccanguyen/Documents/GitHub/H22/ADP.2017-03-27T12_08_50.541.fits" # name of the fits file you want to open
hdu = fits.open(filename) # reads the file
# hdu.info() # prints general info about the file, i.e. number of extensions and their dimensions.
"""
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU    1345   ()
  1  DATA          1 ImageHDU        44   (320, 317, 3682)   float32
  2  STAT          1 ImageHDU        44   (320, 317, 3682)   float32
"""

data = hdu[1].data
hdr = hdu[1].header


#print(data.shape)
"""
(3682, 317, 320)
"""

avg = np.nanmean(data, -1)
spect = np.mean(avg, -1)

# Produce a map/ 2D image

"""
flux_mean = np.nanmean(data,0)
plt.figure(figsize = (7,7)) # setting the size of the image
#plt.title(r"Average flux density in the range $\lambda \in [4750, 9351] \AA{}$")
plt.title("Regions in which we extract spectrum")
im = plt.imshow(np.flip(np.nanmean(data, 0), 0), cmap = "gray", vmin = 0, vmax = 2137)
plt.scatter(164, -160 + 320, label = "(164, 155)")
plt.scatter(82, -239 + 320, label = "(82, 239)")
plt.scatter(150, -100 + 320, label = "(150, 100)")
plt.colorbar(im,fraction = 0.046, pad = 0.04, label = "Flux density [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]")
plt.legend()
plt.savefig("region.png")
plt.show()
"""
lambda0 = hdr['CRVAL3']
dLambda = hdr['CD3_3']
lenWave = hdr['NAXIS3']
wavelengths = np.linspace(lambda0, lambda0 + (lenWave - 1) * dLambda, lenWave)

"""
plt.figure(figsize=(8, 4.5))
plt.plot(wavelengths, spect, lw=1, color='black')
"""


def data_cut(low, up, line):
    wcs = WCS(hdr)[0,:,:] # indexing[spectral, vertical, horizontal]

    lower_boundary = low; upper_boundary = up # upper and lower boundary wavelengths in Angstrom.
    lower_indx = np.array(np.where(wavelengths >= lower_boundary))[0,0] # locating the index
    upper_indx = np.array(np.where(wavelengths <= upper_boundary))[-1,-1]

    extracted_data = data[lower_indx:upper_indx] # cuts the data in the wavelength dimension
    """
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(1,1,1, projection=wcs)
    ax.set_title(f'Average flux density in the range $\lambda \in [{lower_boundary},{upper_boundary}] \AA$')
    im = ax.imshow(np.nanmean(extracted_data,0),cmap="hot", vmin=0,vmax=2137) # no longer needs np.flip because of WCS
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.colorbar(im,fraction=0.046, pad=0.04, label="Flux density [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]")

    plt.savefig(f"{line}")


    plt.figure(figsize=(7, 7))
    plt.title(f'Average flux density in the range $\lambda \in [{lower_boundary},{upper_boundary}] \AA$')
    im = plt.imshow(np.flip(np.nanmean(extracted_data, 0), 0), cmap='hot', vmin=0, vmax=4500)
    plt.colorbar(im, fraction=.046, pad=.04, label='Flux density [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]')
    plt.tight_layout()
    #plt.savefig(f"{line}")
    """
    return extracted_data

dataHalpha = data_cut(6589, 6609, "halphaflux1.png") # Halpha
dataOIII = data_cut(4096, 5016, "OIIIflux1.png")

def mss(data, x, y):
    r = 3
    center_x = x
    center_y = y
    indx = [center_y-r, center_y+r,center_x-r,center_x+r] # [y0, y1 , x0, x1] # aperture indices

    plt.scatter(x, y)
    im = plt.imshow(np.flip(np.nanmean(data, 0), 0), cmap='hot', vmin=0, vmax=4500)


    collapsed = np.nansum(data,0) # summing all non-nan values in the spectral dimension
    aperture_data = collapsed[indx[0]:indx[1], indx[2]:indx[3]]

    mean_flux = np.mean(aperture_data) # taking the mean over both spatial dimensions
    sum_flux = np.sum(aperture_data) # summing up all the flux contained in the aperture.
    std_dev = np.std(aperture_data)

    return mean_flux, sum_flux, std_dev

mean, sum, std = mss(dataHalpha, 50, 150)
print(mean)
print(std)

"""


"""

#plt.show()
