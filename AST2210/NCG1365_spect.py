import numpy as np
import matplotlib.pyplot as plt

nucleus = "/Users/rebeccanguyen/Documents/GitHub/H22/galaxy_nucleus.txt"
arm = "/Users/rebeccanguyen/Documents/GitHub/H22/outer_arm.txt"
arm2 = "/Users/rebeccanguyen/Documents/GitHub/H22/arm2.txt"

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

fig, ax = plt.subplots(3)
ax[0].plot(wavn[1400:1800], fluxn[1400:1800], color = "black", label = "Nucleus of the galaxy", lw = 1)
ax[0].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[0].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[0].axvline(6584.4, color = 'm', label = '[NII]', ls = '--')
ax[0].axvline(6754.2, color = 'g', label = '[SII]', ls = '--')
ax[0].axvline(6768.6, color = 'c', label = '[SII]', ls = '--')
ax[0].set_xticklabels([])
ax[0].set_title(rf"Spectrum from 3 regions, $\lambda")
ax[0].legend()


ax[1].plot(wava[1400:1800], fluxa[1400:1800], color = "black", label = "Upper outer arm", lw = 1)
ax[1].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[1].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[1].axvline(6584.4, color = 'm', label = '[NII]', ls = '--')
ax[1].axvline(6754.2, color = 'g', label = '[SII]', ls = '--')
ax[1].axvline(6768.6, color = 'c', label = '[SII]', ls = '--')
ax[1].set_xticklabels([])
ax[1].legend()

ax[2].plot(wav2[1400:1800], flux2[1400:1800], color = "black", label = "Middle lower arm", lw = 1)
ax[2].axvline(6599.2, color = 'r', label = r'H$\alpha$', ls = '--')
ax[2].axvline(6619.9, color = 'b', label = '[NII]', ls = '--')
ax[2].axvline(6584.4, color = 'm', label = '[NII]', ls = '--')
ax[2].axvline(6754.2, color = 'g', label = '[SII]', ls = '--')
ax[2].axvline(6768.6, color = 'c', label = '[SII]', ls = '--')
ax[2].legend()


plt.show()
