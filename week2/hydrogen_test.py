import astropy.constants as c
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
plt.ion()  # Enable interactive mode for plotting

# Physical constants
a0 = c.a0.to(u.m).value  # Bohr radius in meters
m_e = c.m_e.to(u.kg).value  # Electron mass in kg
hbar = c.hbar.to(u.J * u.s).value  # Reduced Planck constant in J*s
e = c.e.si.value  # Electron charge in Coulombs (SI units)
eps0 = c.eps0.to(u.F/u.m).value  # Vacuum permittivity in SI

# Domain: 0 to 20 Bohr radii
r_max = 20 * a0
r_min = 1e-4*a0  # Avoid zero to prevent singularity in potential
r = np.linspace(r_min, r_max, 50)  #For l=0, we start including r=0.

# Hydrogen atom radial equation for general l
# Schrödinger equation in radial form
# d^2u/dr^2 + (2m/hbar^2)(V-E)u - l(l+1)/r^2 * u = 0
# where V = -e^2/(4πε₀r), 
# and u = r * R(r), with R(r) being the radial wavefunction.
# The potential V is singular at r=0, but the singular
# term S doesn't seem to quite work.

def schrodinger_general(r, y, E, l):
    psi = y[0]
    dpsi = y[1]
    V = -e**2 / (4 * np.pi * eps0 * r)
    d2psi = 2 * m_e / hbar**2 * (V-E) * psi + l * (l + 1) / r**2 * psi    
    return np.vstack((dpsi, d2psi))

# Boundary conditions for general l: u(0)=0, u(r_max)=0
def bc_general(ya, yb, E, l):
	if l==0:
		return np.array([ya[0], yb[0], ya[1]-1])
	else:
	    return np.array([ya[0], yb[0], yb[1]-1])

# Initial guess for wavefunction with orbital angular momentum l
def guess_general(r, l, n):
    # Better initial guess for l=1: r^(l+1) * exp(-r/(n*a0))
    if l == 0:
        psi = r * np.exp(-r / (n * a0))
    elif l == 1:
        psi = r * (-1)**n * np.sin(r / r_max * np.pi * n)/np.pi/n #np.exp(-r /(n*a0))
    else:
        raise UserWarning
    dpsi = np.gradient(psi, r)
    return np.vstack((psi, dpsi))

# Function to find energy eigenvalues by scanning
def find_eigenvalues(l, n_max=10):
    """Find the first n_max eigenvalues for given l"""
    eigenvalues = []
    eigenfunctions = []
    
    # Energy range to scan (expected energies for hydrogen)
    # E_n = -13.6 eV / n^2, where n >= l+1
    n_start = l + 1
    
    for n in range(n_start, n_start + n_max):
        E_guess = (2.5*n - 11) * u.eV.to(u.J)  # Convert to Joules
        
        try:
            sol = solve_bvp(lambda r, y, E: schrodinger_general(r, y, E, l), 
                          lambda ya, yb, E: bc_general(ya, yb, E, l), 
                          r, guess_general(r, l, n), p=[E_guess])
            
            if sol.success:
                eigenvalues.append(sol.p[0])
                eigenfunctions.append(sol)
                print(f"n={n}, l={l}: E = {sol.p[0] / u.eV.to(u.J):.3f} eV")
            
        except Exception as e:
            print(f"Failed to converge for n={n}, l={l}: {e}")
            continue
    
    return eigenvalues, eigenfunctions

# Solve for l=0 (original code)
def schrodinger(r, y, E):
    return schrodinger_general(r, y, E, 0)

def bc(ya, yb, E):
    return bc_general(ya, yb, E, 0)

def guess(r):
    return guess_general(r, 0, 1)

E_guess = (-13.6 * u.eV).to(u.J).value  # Ground state energy in Joules

# Solve for l=0, n=1 (ground state)
sol = solve_bvp(schrodinger, bc, r, guess(r), p=[E_guess])

# Plot l=0 result
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot((sol.x[1:]-r_min) / a0, sol.y[0][1:] / (sol.x[1:]-r_min), label='n=1, l=0')
plt.plot((sol.x[1:]-r_min) / a0, sol.y[0][1:] / a0*2, label='(n=1, l=0) * z')
plt.xlabel('r / a0')
plt.ylabel('ψ(r)')
plt.title('Hydrogen Atom Wavefunction (l=0)')
plt.legend()

# Now solve for the first 10 l=1, m=0 modes
print("\nFinding l=1, m=0 modes:")
r = np.linspace(r_min, r_max, 100)
l1_energies, l1_solutions = find_eigenvalues(l=1, n_max=8)

# Plot first few l=1 modes
plt.subplot(2, 2, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(l1_solutions)))
for i, (sol_l1, color) in enumerate(zip(l1_solutions, colors)):
    if sol_l1.success:
        r_plot = sol_l1.x[1:]
        psi_plot = sol_l1.y[0][1:] / r_plot  # Convert u(r) back to R(r)
        plt.plot((r_plot-r_min) / a0, psi_plot, label=f'n={i+2}, l=1', color=color)

plt.xlabel('r / a0')
plt.ylabel('R(r)')
plt.title('Hydrogen Atom Radial Wavefunctions (l=1)')
plt.legend()

# Plot energy levels
plt.subplot(2, 2, 3)
n_values_l0 = [1]
E_values_l0 = [sol.p[0] / u.eV.to(u.J)]
n_values_l1 = list(range(2, 2 + len(l1_energies)))
E_values_l1 = [E / u.eV.to(u.J) for E in l1_energies]

plt.scatter(n_values_l0, E_values_l0, label='l=0', s=100, marker='o')
plt.scatter(n_values_l1, E_values_l1, label='l=1', s=100, marker='s')
plt.xlabel('Principal quantum number n')
plt.ylabel('Energy (eV)')
plt.title('Energy Levels')
plt.legend()
plt.grid(True)

# Plot probability densities
plt.subplot(2, 2, 4)
# l=0 probability density
r_plot = sol.x[1:]
psi_l0 = sol.y[0][1:] / r_plot
prob_density_l0 = (psi_l0**2) * r_plot**2
plt.plot((r_plot-r_min) / a0, prob_density_l0, label='n=1, l=0', linewidth=2)

# l=1 probability densities (first 3)
for i, sol_l1 in enumerate(l1_solutions):
    if sol_l1.success:
        r_plot = sol_l1.x[1:]
        psi_l1 = sol_l1.y[0][1:] / r_plot
        prob_density_l1 = (psi_l1**2) * r_plot**2
        plt.plot((r_plot-r_min) / a0, prob_density_l1, label=f'n={i+2}, l=1', alpha=0.7)

plt.xlabel('r / a0')
plt.ylabel('r²|R(r)|²')
plt.title('Radial Probability Densities')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary
print(f"\nSummary:")
print(f"l=0, n=1: E = {sol.p[0] / u.eV.to(u.J):.3f} eV")
print(f"Found {len(l1_energies)} l=1 modes:")
for i, E in enumerate(l1_energies):
    print(f"  n={i+2}, l=1: E = {E / u.eV.to(u.J):.3f} eV")
    

