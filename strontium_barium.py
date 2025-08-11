import astropy.constants as c, astropy.units as u
import numpy as np

sigma_Hm = 3E-17*u.cm**2
waves = np.array([421.6, 407.8])*u.nm
g_low = 2
g_high = np.array([2,4]) #Assuming first line is to J=1/2, second to J=3.2
#Einstein coefficients for strong lines
As = np.array([1.27e8,1.42e8])*u.s**-1
nu = c.c / waves

#Compute the Einstein B and cross-sections
B = As * (g_high / g_low) * c.c**2/nu**3/(2*c.h)

# scaled by delta_nu/nu.
sigma_scaled = (B * (c.h/4/np.pi)).to(u.cm**2)
