# Hydrogen atom radial equation for general l
# Schrödinger equation in radial form
# d^2u/dr^2 + (2m/hbar^2)(V-E)u - l(l+1)/r^2 * u = 0
# where V = -e^2/(4πε₀r), 
# and u = r * R(r), with R(r) being the radial wavefunction.

# We can write this in a dimensionless form by defining
# x = r / a₀, where a₀ is the Bohr radius, equal to
# a₀ = 4πε₀hbar² / (m_e e²) in SI units.
# The potential V becomes dimensionless as well. This 
# second order equation becomes:
# d²u/dx² -2u/x - l(l+1)/x² * u = E u

# We can solve this by forming a sparse matrix for the second derivative
# and using a sparse linear algebra solver.

import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import simpson

plt.ion()  # Enable interactive mode for plotting

# Define the matrix for the radial Schrödinger equation
def hydrogen_sparse_matrix(l, r_max, dr=0.04):
    # Create a radial grid, which is regular exluding end
    # points (which are defined to be zero by boundary conditions)
    r = np.arange(dr, r_max, dr)
    n = len(r)
    
    # Create the two main diagonals
    main_diag = 2/dr**2 * np.ones(n) - 2/r + l * (l + 1) / r**2
    off_diag = -np.ones(n-1) / dr**2
    
    # Create the sparse matrix
    A = sp.diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(n, n))
    
    return A, r

if __name__ == "__main__":
    # Parameters
    r_max = 30  # in Bohr radii
    dr = 0.025  # step size in Bohr radii
    nk = 20 # number of eigenvalues to compute
    
    # Store results for all l values. The radius r will be the same for all l.
    all_eigenvalues = {}
    all_eigenvectors = {}
    
    # Loop over l values
    for l in [0, 1]:
        print(f"\nSolving for l={l}...")
        
        # Solve the radial Schrödinger equation
        A, r = hydrogen_sparse_matrix(l, r_max, dr)
        
        # Solve the eigenvalue problem
        eigenvalues, eigenvectors = spla.eigsh(A, k=nk, which='SM')
         
        # Extend the r array and eigenvector array to include the 0 point
        r_extended = np.concatenate(([0], r))  # Include r=0
        eigenvectors_extended = np.row_stack((np.zeros(nk), eigenvectors))
        
        # Normalize the eigenvectors so that ∫|u(r)|² dr = 1
        # where u(r) = r·R(r) is the radial wavefunction
        print(f"  Normalizing eigenvectors for l={l}...")
        for i in range(nk):
            # Calculate the norm using Simpson's rule
            norm = np.sqrt(simpson(eigenvectors_extended[:, i]**2, r_extended))
            eigenvectors_extended[:, i] /= norm
        
        # Store results
        all_eigenvalues[l] = eigenvalues
        all_eigenvectors[l] = eigenvectors_extended
         
        # Plot the first few wavefunctions
        plt.figure(figsize=(10, 6))
        n_start = l + 1  # Principal quantum number starts at l+1
        for i in range(4):  # Plot first 4 computed states
            # Handle division by zero at r=0
            r_plot = r_extended[1:]  # Exclude r=0 point
            wavefunction_plot = eigenvectors_extended[1:, i] / r_plot
            plt.plot(r_plot, wavefunction_plot, label=f'n={i+n_start}, l={l}')
        
        plt.xlabel('r (Bohr radii)')
        plt.ylabel('Wavefunction R(r)')
        plt.title(f'Hydrogen Atom Radial Wavefunctions R(r) (l={l})')
        
        # Print the first few eigenvalues
        print(f"First few eigenvalues for l={l} (in eV):")
        for i in range(4):
            print(f"n={i+n_start}, l={l}: {eigenvalues[i].real * 13.6057:.4f} eV")
    
    # Now let's compute the dipole matrix elements for a transition
    # from l=0 to l=1, which is non-zero only for m=0 
    # (i.e. in the "z" direction, which is r cos(theta) or r mu).
    # The dipole matrix element is given by:
    # <l'=1, m'=0 | \vect{r} | l=0, m=0>
    # = \int R_{l'=1}(r) r R_{l=0}(r) r^2 dr * (1/2) \int_-1^1 mu^2 dmu
    # = \int u_{l'=1}(r) r u_{l=0}(r) dr * (1/3)
    # where mu = cos(theta) is the angular part.
    
    print("\nComputing dipole matrix elements for l=0 -> l=1 transitions...")
    
    # Get the stored results
    eigenvectors_l0 = all_eigenvectors[0]
    eigenvectors_l1 = all_eigenvectors[1]
    
    # Compute the integral using Simpson's rule
    inner_products = np.zeros(nk)

    # Ideal wavefunction for a dipole transition to l=0, n=1
    ideal_wf = r_extended * eigenvectors_l0[:,0]
    #Normalise
    ideal_wf /= np.sqrt(simpson(ideal_wf**2, r_extended))
    # Plot this as a dashed line
    plt.plot(r_extended[1:], ideal_wf[1:]/r_extended[1:], 'k--', label='Ideal l=0, n=1 wavefunction', alpha=0.5)
    # Compute this ideal inner product
    ideal_inner_product = simpson(r_extended * ideal_wf * eigenvectors_l0[:, 0], r_extended)
    
    # Since eigenvectors are now normalized, we can compute dipole elements directly
    for i in range(nk):
        # Radial part: <l'=1 | r | l=0>
        # Both eigenvectors are normalized, so no need for additional normalization
        radial_integral = simpson(r_extended * eigenvectors_l1[:, i] * eigenvectors_l0[:, 0], r_extended)
        inner_products[i] = radial_integral

    print("Dipole matrix elements <l=1,n|r|l=0,n=1>:")
    for i in range(min(4, nk)):
        print(f"  l=1, n={i+2} -> l=0, n=1: {inner_products[i]:.6f}")
    print(f"  Ideal matrix element: {ideal_inner_product:.6f}")
    
    # Finally, form a weighted average of the dipole matrix elements
    # to get the overall transition strength
    mixed_mode = np.sum(inner_products * eigenvectors_l1, axis=1)
    # Now plot the mixed mode, and plot it on the same figure as a dotted line.
    mixed_mode /= np.sqrt(simpson(mixed_mode**2, r_extended))  # Normalize the mixed mode
    plt.plot(r_extended[1:], mixed_mode[1:]/r_extended[1:], 'r:', label='Mixed mode', alpha=0.7)
    plt.legend()
    plt.grid()
    plt.show()