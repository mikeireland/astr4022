"""
Compute the emergent flux from a grey atmosphere
We will approximate q(tau)=0.71044 - 0.1*np.exp(-2.0*tau), 
which is justified by the code grey.py.
"""
from scipy.special import expn
from scipy.integrate import quad
import astropy.constants as c
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# Although you've learnt that using astropy is simple, it is
# also slow. So we will use the SI values of the constants.
# An even better idea is often to use dimensionless units.
planck_C1 = (2*c.h*c.c**2/(1*u.um)**5).si.value
planck_C2 = (c.h*c.c/(1*u.um)/c.k_B/(1*u.K)).si.value

def Flambda_integrand(tau, wave_um, Teff):
	"""
	The Integrand of equaiton 17.70
	"""
	q = 0.71044 - 0.1*np.exp(-2.0*tau)
	T = (0.75*Teff**4*(tau + q))**.25
	return 2*np.pi*planck_C1/wave_um**5/(np.exp(planck_C2/wave_um/T)-1)*expn(2, tau)
	
Teff=5772
nwave = 100
wave_ums = np.linspace(.1,2,nwave)
Fs = np.empty(nwave)
for ix, wave_um in enumerate(wave_ums):
	res = quad(Flambda_integrand, 0, np.inf, args=(wave_um, Teff), full_output=True)
	Fs[ix] = res[0]
	
plt.figure(1)
plt.clf()
plt.plot(wave_ums, Fs/1e6, label='Grey Flux')
plt.plot(wave_ums, np.pi*planck_C1/wave_ums**5/(np.exp(planck_C2/wave_ums/Teff)-1)/1e6, label='Blackbody Flux')
plt.legend()
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.axis([.1,2,0,8.5e7])
plt.tight_layout()

Tmin = Teff * (0.75*0.577)**.25
plt.figure(2)
plt.clf()
plt.plot(wave_ums, Fs/1e6, label='Grey Flux')
plt.plot(wave_ums, np.pi*planck_C1/wave_ums**5/(np.exp(planck_C2/wave_ums/Tmin)-1)/1e6, label='Min T LTE')
plt.legend()
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.axis([.1,2,0,8.5e7])
plt.tight_layout()