#First exercise in lecture...

# Prompt: 
# using astropy.units and astropy.constants, compute the fraction of hydrogen atoms in the 
# n=2 state according to the Boltzmann equation
from astropy.constants import k_B, h, c, G
import numpy as np

import astropy.units as u

# Parameters
T = 5000 * u.K  # Temperature, adjust as needed

# Energy levels for hydrogen (n=1 and n=2)
def hydrogen_energy(n):
    return -13.6 * u.eV / n**2

E1 = hydrogen_energy(1)
E2 = hydrogen_energy(2)

# Statistical weights (degeneracy): g_n = 2n^2
g1 = 2 * 1**2
g2 = 2 * 2**2

# Boltzmann equation: n2/n1 = (g2/g1) * exp(-(E2-E1)/(k_B*T))
delta_E = E2 - E1
boltzmann_fraction = (g2 / g1) * np.exp(-(delta_E / (k_B * T)).decompose().value)

print(f"Fraction of H atoms in n=2 state (n2/n1) at T={T}: {boltzmann_fraction:.3e}")

# Second exercise (end of lecture):
kappa = 1*u.cm**2/u.g
L_on_M = (4*np.pi*G*c/kappa).to(u.L_sun/u.M_sun)
print(f"L/M ratio to create a solar wind: {L_on_M:.3e} L_sun/M_sun")