# Let's compute the structure of an atmosphere, using the folowing modules and assumptions
# 1) A grey atmosphere and hydrostatic equilibrium (analytic)
# 2) An equation of state using the Saha equation: tabulated in rho_Ui_mu_ns_ne.fits
# 3) Opacities computed using the methods in opac.py: tabulated in Ross_Planck_opac.fits
#
# For speed, units are:
# - Length: cm
# - Mass: g
# - Time: s
# - Temperature: K
# - Frequency: Hz

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, cumulative_trapezoid
import astropy.units as u
import astropy.constants as c
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import opac
from scipy.special import expn
from strontium_barium import *
plt.ion()

Teff = 8777 # K 
g = 27400   # cm/s^2

# Set to 1.3 to limit T due to the onset of convection.
# If set to 2.0, there is no effect.
convective_cutoff = 2.0 #1.3

# Load the opacity table for Rosseland mean.
f_opac = pyfits.open('Ross_Planck_opac.fits')
kappa_bar_Ross = f_opac['kappa_Ross [cm**2/g]'].data

# Construct the log(P) and T vectors. 
h = f_opac[0].header
T_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
Ps_log10 = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']

#Create our interpolator functions
f_kappa_bar_Ross = RegularGridInterpolator((Ps_log10, T_grid), kappa_bar_Ross)

def T_tau(tau, Teff):
	"""
	Temperature for a simplified grey atmosphere, with an analytic
    approximation for the Hopf q (feel free to check this!)
	"""
	q = 0.71044 - 0.1*np.exp(-2.0*tau)
	T = (0.75*Teff**4*(tau + q))**.25
	return T

def dPdtau(_, P, T):
	"""
	Compute the derivative of pressure with respect to optical depth.
	"""
	kappa_bar = f_kappa_bar_Ross((np.log10(P), T))
	return g / kappa_bar

# Starting from the lowest value of log(P), integrate P using solve_ivp
#solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
P0 = 10**(Ps_log10[0]) # Initial pressure in dyn/cm^2
tau_grid = np.concatenate((np.arange(3)/3*1e-8,np.logspace(-8,1,60)))
sol = solve_ivp(dPdtau, [0, 10], [P0], args=(Teff,), t_eval=tau_grid, method='RK45')
Ps = sol.y[0]
Ts = T_tau(tau_grid, Teff)
# Artificially cut the deep layer temperature due to convection.
Ts = np.minimum(Ts,convective_cutoff*Teff)

# Load the equation of state
f_eos = pyfits.open('rho_Ui_mu_ns_ne.fits')
rho = f_eos['rho [g/cm**3]'].data
mu = f_eos['mu'].data
ns = f_eos['ns'].data
ne = f_eos['n_e'].data

# Add interpolation functions for the number densities
f_nHI = RegularGridInterpolator((Ps_log10, T_grid), ns[:,:,0])
f_nHII = RegularGridInterpolator((Ps_log10, T_grid), ns[:,:,1])
f_nHm = RegularGridInterpolator((Ps_log10, T_grid), ns[:,:,2])
f_ne = RegularGridInterpolator((Ps_log10, T_grid), ne)
f_rho = RegularGridInterpolator((Ps_log10, T_grid), rho)

# Interpolate onto the tau grid
nHIs = f_nHI((np.log10(Ps), Ts))
nHIIs = f_nHII((np.log10(Ps), Ts))
nHms = f_nHm((np.log10(Ps), Ts))
nes = f_ne((np.log10(Ps), Ts))
kappa_bars = f_kappa_bar_Ross((np.log10(Ps), Ts))
rhos = f_rho((np.log10(Ps), Ts))    

# First, lets plot a continuum spectrum
wave = np.linspace(50, 2000, 1000) * u.nm  # Wavelength in nm
flux = np.zeros_like(wave)  # Initialize flux array

# Just like in grey_flux.py, but in frequency 
planck_C1 = (2*c.h*c.c**2/(1*u.um)**5).si.value
planck_C2 = (c.h*c.c/(1*u.um)/c.k_B/(1*u.K)).si.value

# Planck function, like in grey_flux.py
def Blambda_SI(wave_um, T):
    """
    Planck function in cgs units.
    """
    return planck_C1/wave_um**5/(np.exp(planck_C2/wave_um/T)-1)

print("Computing continuum spectrum")
H = np.zeros(len(wave))  # Initialize H array)
# Compute the flux for each wavelength
for i, w in enumerate(wave):
    # Compute the opacity at this wavelength
    kappa_nu_bars = opac.kappa_cont((c.c/w).to(u.Hz).value, Teff, nHIs, nHIIs, nHms, nes)/rhos

    # Now we need S(tau_nu), i.e. B(tau_nu(tau))
    tau_nu  = cumulative_trapezoid(kappa_nu_bars/kappa_bars, x=tau_grid, initial=0)
    wave_um = w.to(u.um).value
    H[i] = 0.5*np.trapz(Blambda_SI(wave_um, Ts) * expn(2, tau_nu), x=tau_nu)

# Plot the flux and the blackbody approximation
# So far it isn't great... why?
plt.figure(1)
plt.clf()
plt.plot(wave, 4*np.pi*H /1e6, label='Flux')
plt.plot(wave, np.pi*Blambda_SI(wave.to(u.um).value, Teff) / 1e6, label='Blackbody')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.legend()
plt.show()

# Now lets compute the line profile of the Strontium lines
wave = np.linspace(405,425,10000) * u.nm
print("computing Line Spectrum")
H_line = np.zeros(len(wave))  # Initialize H_line array
# Compute the line profile for each wavelength
for i, w in enumerate(wave):
    # Compute the continuum opacity at this wavelength
    kappa_nu_bars = opac.kappa_cont((c.c/w).to(u.Hz).value, Teff, nHIs, nHIIs, nHms, nes)/rhos

    # Add in the line opacity. Approximate n_H=nHIs, and the abundance of
    # Strontium with respect to Hydrogen is 10**(2.83-12)
    # Dopper velocity is sqrt(kT)/mass. Lets say 2 km/s once micro-turbulence is added.
    for line_wave, ss in zip([421.552, 407.771] * u.nm, sigma_scaled):
        #Similar to lecture slides, but with wavelength rathe than frequency.
        line_width_nm = (2*u.km/u.s / c.c) * line_wave
        #Normalise correctly. 
        line_profile = (line_wave/line_width_nm)*(1/np.sqrt(np.pi))* \
            np.exp(-(w - line_wave)**2 / line_width_nm**2) 
        kappa_line = (10**(2.83-12) * nHIs * ss).to(u.cm**2).value * line_profile
        kappa_nu_bars += kappa_line/rhos

    # Now we need S(tau_nu), i.e. B(tau_nu(tau))
    tau_nu  = cumulative_trapezoid(kappa_nu_bars/kappa_bars, x=tau_grid, initial=0)
    wave_um = w.to(u.um).value
    H_line[i] = 0.5*np.trapz(Blambda_SI(wave_um, Ts) * expn(2, tau_nu), x=tau_nu)

#Plot this
plt.figure(2)
plt.clf()
plt.plot(wave, 4*np.pi*H_line / 1e6, label='Flux')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.legend()
plt.show()