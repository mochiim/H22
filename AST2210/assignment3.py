import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image

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


flux_mean = np.nanmean(data,0)
plt.figure(figsize = (7,7)) # setting the size of the image
plt.title(r"Average flux density in the range $\lambda \in [4750, 9351] \AA{}$")
im = plt.imshow(np.flip(np.nanmean(data, 0), 0), cmap = "gray", vmin = 0, vmax = 2137)
plt.scatter(164, -160 + 320, 5, label = "1")
plt.scatter(82, -239 + 320, 5, label = "2")
plt.scatter(150, -100 + 320, 5, label = "3")
#plt.colorbar(im,fraction = 0.046, pad = 0.04, label = "Flux density [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]")
#plt.savefig("2D_img.png")
plt.legend()
plt.show()


lambda0 = hdr['CRVAL3']
dLambda = hdr['CD3_3']
lenWave = hdr['NAXIS3']
wavelengths = np.linspace(lambda0, lambda0 + (lenWave - 1) * dLambda, lenWave)

lower_boundary = 6590; upper_boundary = 6610 # upper and lower boundary wavelengths in Angstrom.
lower_indx = np.array(np.where(wavelengths >= lower_boundary))[0,0] # locating the index
upper_indx = np.array(np.where(wavelengths <= upper_boundary))[-1,-1]

# Collapsed image
"""
plt.figure(figsize=(8, 4.5))
plt.plot(wavelengths, spect, lw=1, color='black')
"""
r = 5
center_x = 170
center_y = 150
indx = [center_y-r, center_y+r,center_x-r,center_x+r] # [y0, y1 , x0, x1] # aperture indices
collapsed = np.nansum(data,0) # summing all non-nan values in the spectral dimension
aperture_data = collapsed[indx[0]:indx[1], indx[2]:indx[3]]
mean_flux = np.mean(aperture_data) # taking the mean over both spatial dimensions
sum_flux = np.sum(aperture_data) # summing up all the flux contained in the aperture.
std_dev = np.std(aperture_data)
