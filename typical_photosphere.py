"""
Lets make plots of the line formation region of a typical photosphere.

The elements we care about are:

elt_names = np.array(['H', 'He',   'C',   'N',   'O',   'Ne',  'Na',  'Mg',  'Si',  'S',   'K',   'Ca',  'Fe'])
    
Ground state:

Na
Ca
Ca+
Fe
Fe+ (strong lines at 23000 cm^-1)



Excited State:

H: From (g/g_0)=4 excited state at 10.2eV. 82269 cm^-1
He: From 159856 cm^-1
Mg: From  21870 cm^-1
Si: From 15394 cm^-1 (390 nm only)


Mg+: From 71491 cm^-1
C: From 61981 cm^-1
N: From 83317
S: From 55330
"""
from saha_eos import *
import astropy.units as u
import astropy.constants as c
from labellines import labelLines
csfont = {'fontname':'Times New Roman'}
plt.ion()

sigma_Hm=3e17
sigma_Hn3=5e18

Eion_Hm = 0.754*u.eV
Hneq2 = 13.6*u.eV*(1-1/4)
Hneq3 = 13.6*u.eV*(1-1/9)
Neneq2 = c.c*c.h*134459*u.cm**(-1)
Heneq2 = c.c*c.h*159855*u.cm**(-1)

Teff = np.linspace(3000,20000,50)
Tphot = Teff * 0.85
expEkT = np.exp(-10*u.eV/(c.k_B*Tphot*u.K))
expEkT0 = np.exp(-10*u.eV/(c.k_B*9000*u.K))
chi_mass = (3e1*expEkT/(expEkT + expEkT0) + 3e-2)*u.cm**2/u.g

#An approximation for the main sequence mass (covered in ASTR2013?)
M_Msun = (Teff/5772)**(4/3)

# Gravity, scaled by the solar value, noting that R \propto M
g = (2.74e4*u.cm/u.s**2) / (M_Msun)

#Now the number density as a function of photospheric temperature
N0 = (g/chi_mass/c.k_B/(Teff*u.K)).cgs

#Convert to density
rho0 = (N0*u.u).cgs

#Plot the density
plt.figure(4)
plt.clf()
plt.semilogy(Teff, N0.value)
plt.axis([4500,15000,1e14,3e18])
plt.xlabel('Teff (K)')
plt.ylabel(r'Photosphere Number Density (cm$^{-3}$)')
plt.tight_layout()
plt.savefig('photosphere_density.pdf')

#Next, lets go through every temperature.
n_e_T = []
ns_T = []
for T, rho in zip(Teff, rho0):
	n_e, ns, mu, Ui  = ns_from_rho_T(rho,T*u.K)
	n_e_T += [n_e.value]
	ns_T += [ns]
ns_T = np.array(ns_T)
n_e_T = np.array(n_e_T)

abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = solarmet()
nelt = len(elt_names)

#Now explicitly compute what is needed for continuum opacities
wave_debroglie = np.sqrt(c.h**2/2/np.pi/c.m_e/c.k_B/(Teff*u.K)).cgs.value
n_Hm = ns_T[:,0]*n_e_T*wave_debroglie**3/2*1/2*np.exp(-Eion_Hm/c.k_B/(Teff*u.K))
n_Hneq3 = 9*ns_T[:,0] * np.exp(-Hneq3/c.k_B/(Teff*u.K))
n_Hneq2 = 4*ns_T[:,0] * np.exp(-Hneq2/c.k_B/(Teff*u.K))
n_Heneq2 = 12*ns_T[:,2] * np.exp(-Heneq2/c.k_B/(Teff*u.K))
n_Neneq2 = 12*ns_T[:,14] * np.exp(-Neneq2/c.k_B/(Teff*u.K))

#'H', 'He',   'C',   'N',   'O',   'Ne',  'Na',  'Mg',  'Si',  'S',   'K',   'Ca',  'Fe']
#A plot of the number densities of the neutral species.
n_p = ns_T[:,0] + ns_T[:,1]
plt.figure(1)
plt.clf()
ax = plt.axes()
order = np.argsort(ns_T[0,3*np.arange(1,nelt)-1]/n_p[0])
for i in range(nelt-1):
	ax.semilogy(Teff, ns_T[:,3*order[i]+2]/n_p, label=elt_names[order[i]+1] + 'I')
ax.semilogy(Teff, ns_T[:,0]/n_p, label='HI')
ax.axis([4500,15000,10**(-10.8),10**0.5])
plt.ylabel(r'N/N$_H$')
plt.xlabel(r'T$_{\rm eff}$')
labelLines(ax.get_lines(), zorder=2.5, **csfont)
plt.tight_layout()

#A plot of the number density of ionized species.
plt.figure(2)
plt.clf()
ax = plt.axes()
order = np.argsort(ns_T[-1,3*np.arange(1,nelt)-1]/n_p[0])
for i in range(nelt-1):
	ax.semilogy(Teff, ns_T[:,3*order[i]+3]/n_p, label=elt_names[order[i]+1] + 'II')
	
ax.axis([4500,15000,3e-8,.1])
plt.ylabel(r'N/N$_H$')
plt.xlabel(r'T$_{\rm eff}$')
labelLines(ax.get_lines(), zorder=2.5, **csfont)
plt.tight_layout()

#Make a plot of the model continuum opacity.
plt.figure(3)
plt.clf()
#Number densities in cgs units multiplied by the cross sections in cgs units.
chi_mass2 = (n_Hneq3*5e-18*u.cm**(-1)+ n_Hm*3.1e-17*u.cm**(-1) + n_e_T*6.65e-25*u.cm**(-1))/rho0 
plt.semilogy(Teff, chi_mass2.cgs.value, label='Grey Opacity of H, H-, e-')
plt.semilogy(Teff, chi_mass.cgs.value, label='Simple formula above')
plt.axis([4000,14000,1e-2,40])
plt.xlabel(r'T$_{\rm eff}$ (K)')
plt.ylabel(r'$\bar{\chi}$ (cm$^2$/g)')
plt.tight_layout()
plt.legend()

# Now a plot of number densities of key elements and excited states that
# contribute to continuum opacity.
plt.figure(5)
plt.clf()
ax = plt.axes()
order = np.argsort(ns_T[0,3*np.arange(1,nelt)-1]/n_p[0])
ax.semilogy(Teff, n_Hm/n_p, 'k--', label='H-')
for i in range(nelt-1):
	if elt_names[order[i]+1] in ['He', 'Ca', 'K', 'Na']:
		ax.semilogy(Teff, ns_T[:,3*order[i]+2]/n_p, label=elt_names[order[i]+1] + 'I')
ax.semilogy(Teff, ns_T[:,0]/n_p, label='HI')
ax.semilogy(Teff, n_Hneq3/n_p, 'b:', label='HI n=3')
ax.semilogy(Teff, n_Hneq2/n_p, 'g:', label='HI n=2')
ax.semilogy(Teff, n_Heneq2/n_p, 'k:', label='HeI n=2')
ax.semilogy(Teff, n_e_T/n_p, 'm:', label='e-')
ax.semilogy(Teff, n_e_T/n_p/1e7, 'm--', label=r'e- /10^7 ')

ax.axis([3000,20000,10**(-10.8),10**0.5])
plt.ylabel(r'N/N$_H$')
plt.xlabel(r'T$_{\rm eff}$')
#plt.legend()
labelLines(ax.get_lines(), zorder=2.5, **csfont)
plt.tight_layout()
