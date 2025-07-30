import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import scipy.optimize as op
import time
#For fsolve - no idea why this happens at high temperatures...
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

def solarmet():
    """Return solar metalicity abundances by number and masses for low mass elements.
    From Asplund et al (2009), up to Oxygen only"""
    abund = 10**np.array([0.00,-1.07,-10.95,-10.62,-9.3,-3.57,-4.17,-3.31])
    masses=  np.array([1.0, 4.0, 6.94, 9.01, 10.81, 12.01, 14.01, 16.00])
    elt_names = np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O'])
    return abund, masses, elt_names
    
def saha(n_e, T):
    """Compute the solution to the Saha equation as a function of electron number
    density and temperature. This enables the problem to be a simple Ax=b linear problem.
    Results from this function can be used to solve the Saha equation as e.g. a function 
    of rho and T via e.g. tabulating or solving.
    
    Parameters
    ----------
    n_e: the dimensioned electron number density
    T: Temperature in K.
    
    Returns
    -------
    rho: astropy.units quantity compatible with density
    mu: Mean molecular weight (dimensionless, i.e. to be multiplied by the AMU)
    ns: A vector of number densities of H, H+, He, He+, He++
    
    """
    #Input the abundances of the elements
    abund, masses, _ = solarmet()
    
    #This will break for very low temperatures. In this case, fix a zero  
    #ionization fraction
    if (T<1500*u.K):
        ns = n_e*1e15*np.array([abund[0],0,abund[1],0,0])
    else:
        #The thermal de Broglie wavelength
        debroglie=np.sqrt(c.h**2/2/np.pi/c.m_e/c.k_B/T)
    
        #Hydrogen ionization. We neglect the excited states because
        #they are only important when the series diverges... 
        h1  = 2./debroglie**3 *1/2*np.exp(-13.6*u.eV/c.k_B/T)  
    
        #Helium ionization. NB excited states are still nearly ~20ev higher.
        he1 = 2./debroglie**3 *2/1*np.exp(-24.580*u.eV/c.k_B/T) 
    
        #Helium double-ionization
        he2 = 2./debroglie**3 *1/2*np.exp(-54.403*u.eV/c.k_B/T)
    
        #Now construct our matrix of 5 equations defining these number densities.
        A = np.zeros( (5,5) );
        A[0,0:2]=[-h1/n_e,1]
        A[1,2:4]=[-he1/n_e,1]
        A[2,3:5]=[-he2/n_e, 1]
        A[3,:]  =[abund[1],abund[1],-abund[0],-abund[0],-abund[0]]
        A[4,:] =[0,1,0,1,2]
        
        #Convert the electron density to a dimensionless value prior to solving.
        b=[0,0,0,0,n_e.to(u.cm**(-3)).value]
        ns =np.linalg.solve(A,b)*u.cm**(-3)
    
    #The next lines ensure ionization at high T, due to nuclei being separated by less. 
    #than the Debye length. Somewhat of a hack, but eventually the Saha equation does 
    #break down...
    ns_highT=[0,abund[0],0,0,abund[1]]
    ns_highT=ns_highT/(abund[0]+2*abund[1])*n_e
    if (T > 2e6*u.K):
        ns=ns_highT
    elif (T > 1e6*u.K):
        frac=(T.to(u.K).value-1e6)/1e6
        ns = frac*ns_highT + (1-frac)*ns
        
    #For normalization... we need the number density of Hydrogen
    #nuclei, which is the sum of the number densities of H and H+.
    n_h = np.sum(ns[:2])
    
    #Density. Masses should be scalars.
    rho = n_h*np.sum(abund*masses)*u.u
    
    #Fractional "abundance" of electrons.
    f_e = n_e/n_h
    
    #mu is mean "molecular" weight, and we make the approximation that
    #electrons have zero weight.
    mu = np.sum(abund*masses)/(np.sum(abund) + f_e)
    
    #Finally, we should compute the internal energy with respect to neutral gas.
    #This is the internal energy per H atom, divided by the mass in grams per H atom. 
    Ui=(ns[1]*13.6 + ns[3]*24.58 + ns[4]*(54.403+24.58))*u.eV/n_h/np.sum(abund*masses*u.u);
    
    return rho, mu, Ui, ns
    
def saha_solve(log_n_e_mol_cm3, T, rho_0_in_g_cm3):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value/u.cm**3
    rho, mu, Ui, ns = saha(n_e, T)
    
    return np.log(rho_0_in_g_cm3/rho.to(u.g/u.cm**3).value)
 
def saha_solve_P(log_n_e_mol_cm3, T, P_in_dyn_cm2):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed pressure."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value/u.cm**3
    rho, mu, Ui, ns = saha(n_e, T)
    P = (np.sum(ns)*c.k_B*T).cgs
    #Add the electron pressure to the total pressure.
    P += (n_e*c.k_B*T).cgs
    
    return np.log(P_in_dyn_cm2/P.to(u.dyn/u.cm**2).value)

def rho_from_PT(P, T):
    """Convenience function - Obtain the density from pressure and temperature.
    """
    #Start with the electron number density equal in mol/cm^3 equal to the pressure
    #divided by kT, or a much lower number at low temperatures. Modify it a little to help
    #starting point at low T
    x0 = np.log( (P/c.k_B/T).to(1/u.cm**3).value/c.N_A.value )
    x0 += np.log(2/(np.exp(50e3/T.to(u.K).value) + 1))
    
    #Solve the equation for electron number density logarithm.
    res = op.fsolve(saha_solve_P, x0, args=(T, P.to(u.dyn/u.cm**2).value), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value/u.cm**3
    rho, mu, Ui, ns = saha(n_e, T)

    return rho.to(u.g/u.cm**3)
    
if __name__=='__main__':
	#Saha Test - number density of electrons, and temperature!
    rho, mu, Ui, ns = saha(1e16*u.cm**(-3), 10000*u.K)
    
    #Now the calculations
    abund, masses, elt_names = solarmet()
    np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
    print("Hydrogen Ionisation Fraction: {:.4f}".format(ns[1]/(ns[0] + ns[1])))
    print("Helium Single Ionisation Fraction: {:.4f}".format(ns[3]/(ns[2] + ns[3] + ns[4])))
    print("Helium Double Ionisation Fraction: {:.4f}".format(ns[4]/(ns[2] + ns[3] + ns[4])))
    print("Density (g/cm^3): {:.2e}".format(rho.to(u.g/u.cm**3).value))
    print("Mean Molecular Weight: {:.2f}".format(mu))