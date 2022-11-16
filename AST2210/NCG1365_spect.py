import numpy as np
import matplotlib.pyplot as plt

nucleus = "/Users/rebeccanguyen/Documents/GitHub/H22/nucleus1.txt"
arm = "/Users/rebeccanguyen/Documents/GitHub/H22/upperarm.txt"
arm2 = "/Users/rebeccanguyen/Documents/GitHub/H22/lowerarm.txt"

def read_file(datafile):
    wav = [] # wavelengths [Å]
    flux = [] # flux density
    with open(datafile) as infile:
        for line in infile:
            separate = line.split() # separate elements
            wav.append(eval(separate[0]))
            flux.append(eval(separate[1]))
    return np.array(wav), np.array(flux)

wavn, fluxn = read_file(nucleus)
wava, fluxa = read_file(arm)
wav2, flux2 = read_file(arm2)
"""
fig, ax = plt.subplots(3, figsize=(7, 5))
fig.suptitle(f"Spectrum from 3 regions ", fontsize=16)
ax[0].plot(wavn, fluxn, color = "black", label = "Nucleus, (164, 155)px", lw = 1)
ax[1].plot(wava, fluxa, color = "black", lw = 1, label = "Upper arm, 82, 239)px")
ax[2].plot(wav2, flux2, color = "black", lw = 1, label = "Lower arm, (150, 100)px")
ax[0].legend()
ax[1].legend()
ax[2].legend()
fig.savefig("totspec.png")

fig, ax = plt.subplots(3, figsize=(7, 5))
fig.suptitle(f"Spectrum from 3 regions, $\lambda \in$[{wavn[1400]:.1f}, {wavn[1800]:.1f}] Å", fontsize=16)
ax[0].plot(wavn[1400:1800], fluxn[1400:1800], color = "black", label = "Nucleus, (164, 155)px", lw = 1)
ax[0].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[0].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[0].axvline(6584.4, color = 'b', ls = '--')
ax[0].axvline(6754.2, color = 'g', label = '[SII]', ls = '--')
ax[0].axvline(6768.6, color = 'g', ls = '--')





ax[1].plot(wava[1400:1800], fluxa[1400:1800], color = "black", lw = 1)
ax[1].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[1].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[1].axvline(6584.4, color = 'b', ls = '--')
ax[1].axvline(6754.2, color = 'g', ls = '--')
ax[1].axvline(6768.6, color = 'g', label = '[SII]', ls = '--')
ax[1].legend()
ax[1].set_ylabel("Flux [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]")
ax[1].yaxis.set_label_coords(-.12, .5)


ax[2].plot(wav2[1400:1800], flux2[1400:1800], color = "black", lw = 1, label = "Lower arm, (150, 100)px")
ax[2].set_xlabel("Wavelength [Å]")
ax[2].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[2].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[2].axvline(6584.4, color = 'b', ls = '--')
ax[2].axvline(6754.2, color = 'g', label = '[SII]', ls = '--')
ax[2].axvline(6768.6, color = 'g', ls = '--')


fig.savefig("spectra_highwave.png")
 #####
"""
fig, ax = plt.subplots(3, figsize=(7, 5))
fig.suptitle(f"Spectrum from 3 regions, $\lambda \in$[{wavn[0]:.1f}, {wavn[500]:.1f}] Å", fontsize=16)
ax[0].plot(wavn[:500], fluxn[:500], color = "black", lw = 1, label = "(164, 155)px")
ax[0].axvline(4888.2, color = "c", ls = '--', label = r"H$\beta$")
ax[0].axvline(5034.6, color = "m", ls = '--', label = "[OIII]")
ax[0].axvline(4986.4, color = "m", ls = '--')
ax[0].axvline(5226.4, color = "darkorange", ls = '--', label = "Fe II")


ax[1].plot(wava[:500], fluxa[:500], color = "black", lw = 1)
ax[1].axvline(4888.2, color = "c", ls = '--', label = r"H$\beta$")
ax[1].axvline(5034.6, color = "m", ls = '--', label = "[OIII]")
ax[1].axvline(4986.4, color = "m", ls = '--')
ax[1].axvline(5226.4, color = "darkorange", ls = '--', label = "Fe II")
ax[1].set_ylabel("Flux [$10^{-20}$erg s$^{-1}$cm$^{-2}\AA{}^{-1}$]")
ax[1].yaxis.set_label_coords(-.12, .5)
ax[1].legend()


ax[2].plot(wav2[:500], flux2[:500], color = "black", lw = 1, label = "(150, 100)px")
ax[2].axvline(4888.2, color = "c", ls = '--', label = r"H$\beta$")
ax[2].axvline(5034.6, color = "m", ls = '--', label = "[OIII]")
ax[2].axvline(4986.4, color = "m", ls = '--')
ax[2].axvline(5226.4, color = "darkorange", ls = '--', label = "Fe II")
ax[2].set_xlabel("Wavelength [Å]")



fig.savefig("spectra_wavelow.png")

plt.show()
