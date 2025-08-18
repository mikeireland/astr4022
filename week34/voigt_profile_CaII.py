from scipy.special import voigt_profile
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
plt.ion()

#Now lets make a frequency grid, based on a wavelength grid.
wave = np.linspace(392,395,10000)*u.nm
nu = (c.c/wave).cgs

#Central frequency of our transition
nu0 = (c.c/(3933*u.AA)).cgs

#Gamma from NIST
Gamma = 1.57e8*u.Hz

#Doppler \Delta \nu
Dnu_Doppler = (np.sqrt(2*c.k_B*(6500*u.K)/(40*u.u))/c.c*nu0).cgs

#Lets compute the profile!
phi = voigt_profile((nu-nu0).to(u.Hz).value, (Dnu_Doppler/np.sqrt(2)).to(u.Hz).value, Gamma.to(u.Hz).value/4/np.pi)
phi = phi*u.s
plt.figure(3)
plt.clf()
plt.semilogy(wave,phi)
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Frequency-normalised line profile (Hz$^{-1}$)')

plt.figure(4)
plt.clf()
plt.plot(wave,phi)
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Frequency-normalised line profile (Hz$^{-1}$)')
plt.axis([393.2,393.4,0,np.max(phi).value])


#From Malika and others - to check!
Blu = 4.5e10*u.s/u.g
cross_sect = (Blu * c.h*nu/4/np.pi*phi).cgs