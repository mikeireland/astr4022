from scipy.special import voigt_profile
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import time
import astropy.units as u
import astropy.constants as c
plt.ion()

#The text states that H has integral sqrt(pi)
#As a first test, lets see what the integral of voigt_profile is!
integral, err = voigt_integral = quad(voigt_profile, -np.inf, np.inf, args=(1,1), epsabs=1e-5, epsrel=1e-5)
print('voigt_profile integral: {:.4f}'.format(integral))
#...so it seems we'll have to use the normalised version of H, divided by sqrt(pi)
#This is better anyway, as it is explicitly \phi (the line profile)

#The textbook function H
def H_integrand(y,a,v):
	return a/np.pi*np.exp(-y**2)/((v-y)**2 + a**2)

#A simple test, with a=1 
#(*** at least change the next 2 numbers to change the test ***)
gamma = .5*4*np.pi
delta_nu = 1.0
a = gamma/4/np.pi/delta_nu
x = np.linspace(-20,20,100)

#Here is a super-naieve way to call the scipy function
scipy_y_naieve = voigt_profile(x, delta_nu, gamma)

#Although scipy calls the 3rd argument "gamma", it defines it as:
# 'The half-width at half-maximum of the Cauchy distribution part'
# and sigma is the standard deviation, not quite \Delta \nu.
#Maybe this will work? Time the function as well
scipy_start = time.time()
scipy_y = voigt_profile(x, delta_nu/np.sqrt(2), gamma/4/np.pi)
scipy_end = time.time()

#Now for the explicit integral
explicit_start = time.time()
our_y = np.empty_like(x)
for i, xval in enumerate(x):
	integral, err = quad(H_integrand, -np.inf, np.inf, args=(a, xval/delta_nu))
	our_y[i] = integral
our_y /= np.sqrt(np.pi)
explicit_end = time.time()
print("Line profile PDF sum: {:.3f}".format(np.trapz(our_y, x)))

#Plot the different profiles
plt.figure(1)
plt.clf()
plt.plot(x, our_y, label='Explicit')
plt.plot(x, scipy_y, label='Scipy')
plt.plot(x, scipy_y_naieve, label='Naieve Scipy')
plt.legend()
plt.xlabel('v')
plt.ylabel(r'$\phi$')

#Plot the difference between the scipy and explicit integrals
plt.figure(2)
plt.clf()
plt.plot(x, our_y - scipy_y)
plt.xlabel('v')
plt.ylabel(r'$\Delta\phi$')

#Finally, lets see which was quicker...
print("Scipy time for {:d} points   : {:.6f}s".format(len(x), scipy_end-scipy_start))
print("Explicit time for {:d} points: {:.6f}s".format(len(x), explicit_end-explicit_start))

#Lesson - generally best to used functions someone else has packaged! 
#(once you understand how)
