"""
Here is an extract from some of Mike's equation of state code, focusing on the 
Saha equation.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import scipy.optimize as op
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from astropy.table import Table
#For fsolve - no idea why this happens at high temperatures...
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

#Some constants! For speed, not readability.
debroglie_const = (c.h**2/2/np.pi/c.m_e/c.k_B).cgs.value
eV_Boltzmann_const = (u.eV/c.k_B).cgs.value
deybe_const = (c.k_B/8/np.pi/c.e.esu**2).cgs.value
delchi_const = (c.e.esu**2/(1*u.eV)).cgs.value

def solarmet():
    """Return solar metalicity abundances by number and masses for low mass elements.
    From Asplund et al (2009), up to an abundance of 1e-5 only plus Na, K, Ca. 
    Degeneracy and ionization energies from IA05 in Scholz code"""
    elt_names = np.array(['H', 'He',   'C',   'N',   'O',   'Ne',  'Na',  'Mg',  'Si',  'S',   'K',   'Ca',  'Fe'])
    n_p =        np.array([1,   2,      6,     7,     8,     10,    11,    12,    14,    16,    19,    20,    26])
    masses=      np.array([1.0, 4.0,    12.01, 14.01, 16.00, 18.0,  22.99, 24.31, 28.09, 32.06, 39.10, 40.08, 55.85])
    abund = 10**(np.array([12, 10.93,   8.43,  7.83,  8.69,  7.93,  6.24,  7.60,  7.51,  7.12,  5.03,  6.34,  7.50])-12)
    ionI  =      np.array([13.595,24.58,11.26, 14.53, 13.61, 21.56, 5.14,  7.644, 8.149, 10.357,4.339, 6.111, 7.87])
    ionII  =     np.array([0,  54.403,  24.376,29.593,35.108,40.96, 47.29, 15.03, 16.34, 23.40, 31.81, 11.87, 16.18])

    #Degeneracy of many of these elements are somewhat temperature-dependent,
    #as it is really a partition function. But as this is mostly H/He plus 
    #other elements as a mass reservoir and source of low-T
    #electrons, we're ignoring this. 
    gI =   np.array([2,1,9,4,9,1,2,1,9,9,2,1,25])
    gII =  np.array([1,2,6,9,4,6,1,2,6,4,1,2,30])

    #A lot of these degeneracies are educated guesses! But we're not worried
    #about most elements in the doubly ionized state. 
    gIII = np.array([0,1,1,6,9,9,6,1,1,9,6,1,30])

    return abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names
        
def saha(n_e, T):
    """Compute the solution to the Saha equation as a function of electron number
    density and temperature, in CGS units. 
    
    This enables the problem to be a simple Ax=b linear problem.
    Results from this function can be used to solve the Saha equation as e.g. a function 
    of rho and T via e.g. tabulating or solving.
    
    Parameters
    ----------
    n_e: the dimensioned electron number density in cm^-3
    T: Temperature in K.
    
    Returns
    -------
    rho: astropy.units quantity compatible with density
    mu: Mean molecular weight (dimensionless, i.e. to be multiplied by the AMU)
    ns: A vector of number densities of H, H+, ... He, He+, He++ ...
    
    """
    
    #Input the abundances of the elements
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names  = solarmet()
    n_elt = len(n_p)
    
    #Find the Deybe length, and the decrease in the ionization potential in eV, 
    #according to Mihalas 9-178
    deybe_length = np.sqrt(deybe_const*T/n_e)
    z1_delchi = delchi_const/deybe_length
    
    #This will break for very low temperatures. In this case, fix a zero  
    #ionization fraction
    if (T<1000):
        ns = np.zeros(n_elt*3 - 1)
        ns[0] = abund[0]
        ns[2 + 3*np.arange(n_elt-1)] = abund[1+np.arange(n_elt-1)]
        ns = n_e*1e15*ns
    else:
        #The thermal de Broglie wavelength. See dimensioned version of 
        #this constant above
        debroglie=np.sqrt(debroglie_const/T)
    
        #Hydrogen ionization. We neglect the excited states because
        #they are only important when the series diverges... 
        h1  = 2./debroglie**3 *gII[0]/gI[0]*np.exp(-(ionI[0] - n_p[0]*z1_delchi)*eV_Boltzmann_const/T)  
    
        #Now construct our matrix of n_elt*3-1 equations defining these number densities.
        A = np.zeros( (3*n_elt-1,3*n_elt-1) );
        A[0,0:2]=[-h1/n_e,1]
        for i in range(1,n_elt):
            #Element ionization. NB excited states are still nearly ~20ev higher for He.
            he1 = 2./debroglie**3 *gII[i]/gI[i]*np.exp(-(ionI[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T) 
    
            #Element double-ionization
            he2 = 2./debroglie**3 *gIII[i]/gII[i]*np.exp(-(ionII[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)

            A[3*i-2,3*i-1:3*i+1]=[-he1/n_e, 1]
            A[3*i-1,3*i:3*i+2]  =[-he2/n_e, 1]
            A[3*i,:2]  =[abund[i],abund[i]]
            A[3*i,3*i-1:3*i+2] = [-abund[0],-abund[0],-abund[0]]
        A[-1,:] =np.concatenate(([0,1], np.tile([0,1,2], n_elt-1)))
        
        #Convert the electron density to a dimensionless value prior to solving.
        b =np.zeros((3*n_elt-1))
        b[-1] = n_e
        ns =np.linalg.solve(A,b)
        ns = np.maximum(ns,1e-6)
        #import pdb; pdb.set_trace()
    
    #The next lines ensure ionization at high electron pressure, roughly due to nuclei 
    #being separated by less than the size of an atom. 
    #There is also a hack included based on a typical atom size.
    ns_highT = np.zeros(n_elt*3 - 1)
    ns_highT[1 + np.arange(n_elt)*3] = abund
    ns_highT=ns_highT/(abund[0]+2*np.sum(abund[1:]))*n_e
    atom_size = 1e-8 #In cm
    if (n_e*atom_size**3 > 2):
        ns=ns_highT
        print("High T")
    elif (n_e*atom_size**3 > 1):
        frac=((n_e*atom_size **3) - 1)/1.0
        ns = frac*ns_highT + (1-frac)*ns
        
    #For normalization... we need the number density of Hydrogen
    #nuclei, which is the sum of the number densities of H and H+.
    n_h = np.sum(ns[:2])
    
    #Density. Masses should be scalars.
    rho_cgs = n_h*np.sum(abund*masses)*c.u.to(u.g).value
   
    #Fractional "abundance" of electrons.
    f_e = n_e/n_h
    
    #mu is mean "molecular" weight, and we make the approximation that
    #electrons have zero weight.
    mu = np.sum(abund*masses)/(np.sum(abund) + f_e)
    
    #Finally, we should compute the internal energy with respect to neutral gas.
    #This is the internal energy per H atom, divided by the mass in grams per H atom. 
    Ui=(ns[1]*13.6 + ns[3]*24.58 + ns[4]*(54.403+24.58))*u.eV/n_h/np.sum(abund*masses*u.u);
    
    return rho_cgs, mu, Ui, ns
    
def saha_solve(log_n_e_mol_cm3, T, rho_0_in_g_cm3):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T)
    
    return np.log(rho_0_in_g_cm3/rho)
 
def ns_from_rho_T(rho,T):
    """Compute number densities given a density and temperature
    
    Parameters
    ----------
    rho: density in g/cm^-3
    T: Temperature in K.
    
    Returns
    -------
    Electron number density, element & ion number densities, mean molecular weight
    and internal energy.
    """
    
    rho_in_g_cm3 = rho.to(u.g/u.cm**3).value
    
    #Start with the electron number density equal in mol/cm^3 equal to the density
    #in g/cm^3, or a much lower number at low temperatures. Modify it a little to help
    #starting point at low T
    x0 = np.log(rho_in_g_cm3)
    T_K = np.maximum(T.to(u.K).value,1000)
    x0 += np.log(2/(np.exp(50e3/T_K) + 1))
    
    #The following line is the important one that can't have units associated
    #with it, as it takes too long.
    res = op.fsolve(saha_solve, x0, args=(T.cgs.value, rho_in_g_cm3), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value
    rho_check, mu, Ui, ns = saha(n_e, T.cgs.value)
    if (np.abs(rho_check/rho_in_g_cm3-1) > 0.01):
        raise UserWarning("Density check incorrect!")

    #Return dimensioned quantities
    return n_e*(u.cm**(-3)), ns*(u.cm**(-3)), mu, Ui 
 
if __name__=='__main__':
	#Saha Test - number density of electrons, and temperature!
    rho, mu, Ui, ns = saha(1e18, 50000)
    
    #Now the calculations
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names  = solarmet()
    np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
    print("Hydrogen Ionisation Fraction: {:.4f}".format(ns[1]/(ns[0] + ns[1])))
    print("Neutral, 1st ionisation, 2nd ionisation fraction of other elements: ")
    for i in range(len(elt_names)-1):
        print("{:2s}:".format(elt_names[i+1]) + str(ns[2+3*i:5+3*i]/np.sum(ns[0:2])/abund[i+1]))
    print("Density (g/cm^3): {:.2e}".format(rho))
        
