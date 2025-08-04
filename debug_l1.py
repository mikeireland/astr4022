import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import astropy.constants as c
import astropy.units as u

plt.ion()

# Physical constants in SI units
a0 = c.a0.to(u.m).value  # Bohr radius
m_e = c.m_e.to(u.kg).value  # Electron mass
hbar = c.hbar.to(u.J * u.s).value  # Reduced Planck constant
e = c.e.si.value  # Elementary charge
eps0 = c.eps0.to(u.F/u.m).value  # Vacuum permittivity

def debug_l1_modes():
    """Debug why l=1 modes are not being found"""
    
    # Set up radial grid
    r_max = 50 * a0  # 50 Bohr radii
    r_min = 1e-12 * a0  # Small but not zero
    
    def radial_schrodinger(r, y, E, l):
        """Radial Schrödinger equation"""
        u, du_dr = y
        V = -e**2 / (4 * np.pi * eps0 * r)
        d2u_dr2 = (2 * m_e / hbar**2) * (V - E) * u + l * (l + 1) / r**2 * u
        return [du_dr, d2u_dr2]
    
    def solve_radial_equation(E, l, r_span, n_points=1000):
        """Solve the radial equation for given energy E and angular momentum l"""
        r_eval = np.linspace(r_span[0], r_span[1], n_points)
        r0 = r_span[0]
        
        if l == 1:
            y0 = [r0**2, 2*r0]  # u ≈ r^2, du/dr ≈ 2r  
        else:
            y0 = [r0**(l+1), (l+1) * r0**l]
        
        sol = solve_ivp(radial_schrodinger, r_span, y0, args=(E, l), 
                       t_eval=r_eval, dense_output=True, rtol=1e-10, atol=1e-12)
        
        if sol.success:
            return sol.t, sol.y[0]  # Return r, u(r)
        else:
            raise RuntimeError("ODE solver failed")
    
    def boundary_function(E, l):
        """Function whose zeros are the eigenvalues"""
        try:
            r_vals, u_vals = solve_radial_equation(E, l, (r_min, r_max))
            return u_vals[-1]  # u(r_max) should be zero
        except:
            return np.inf
    
    # Test specific energies for l=1
    print("Testing l=1 boundary function at theoretical energies...")
    
    # Theoretical l=1 energies (n=2,3,4,...)
    theoretical_energies = [-13.6/n**2 for n in range(2, 8)]
    
    for i, E_theory_eV in enumerate(theoretical_energies):
        E_theory = E_theory_eV * u.eV.to(u.J)
        
        try:
            bc_val = boundary_function(E_theory, 1)
            print(f"  n={i+2}: E = {E_theory_eV:.3f} eV, boundary = {bc_val:.6e}")
            
            # Also test nearby energies
            for factor in [0.99, 1.01]:
                E_test = E_theory * factor
                E_test_eV = E_test / u.eV.to(u.J)
                bc_test = boundary_function(E_test, 1)
                print(f"    {factor*100:.0f}% of theory: E = {E_test_eV:.3f} eV, boundary = {bc_test:.6e}")
                
        except Exception as e:
            print(f"  n={i+2}: Failed to evaluate boundary function: {e}")
    
    # Plot boundary function over energy range
    print("\nPlotting boundary function...")
    
    # Energy range for l=1
    E_min = -4.0 * u.eV.to(u.J)
    E_max = -0.1 * u.eV.to(u.J)
    E_values = np.linspace(E_min, E_max, 500)
    
    boundary_values = []
    E_plot = []
    
    for E in E_values:
        try:
            bc = boundary_function(E, 1)
            if np.isfinite(bc) and abs(bc) < 1e10:  # Filter out extremely large values
                boundary_values.append(bc)
                E_plot.append(E / u.eV.to(u.J))
        except:
            continue
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(E_plot, boundary_values, 'b-', linewidth=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Boundary function u(r_max)')
    plt.title('Boundary function for l=1')
    plt.grid(True, alpha=0.3)
    
    # Mark theoretical energies
    for i, E_theory_eV in enumerate(theoretical_energies[:5]):
        if E_min/u.eV.to(u.J) <= E_theory_eV <= E_max/u.eV.to(u.J):
            plt.axvline(x=E_theory_eV, color='g', linestyle=':', alpha=0.7, 
                       label=f'n={i+2}' if i < 3 else "")
    
    plt.legend()
    
    # Zoom in on the region around n=2
    plt.subplot(2, 1, 2)
    E_zoom_min = -4.0
    E_zoom_max = -3.0
    E_zoom_indices = [i for i, E in enumerate(E_plot) if E_zoom_min <= E <= E_zoom_max]
    
    if E_zoom_indices:
        E_zoom = [E_plot[i] for i in E_zoom_indices]
        bc_zoom = [boundary_values[i] for i in E_zoom_indices]
        
        plt.plot(E_zoom, bc_zoom, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=-3.4, color='g', linestyle=':', alpha=0.7, label='Theory n=2')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Boundary function u(r_max)')
        plt.title('Zoom: Region around n=2 (l=1)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Try a different approach: plot the wavefunction at theoretical energy
    print("\nPlotting wavefunction at theoretical n=2, l=1 energy...")
    
    E_n2_l1 = -3.4 * u.eV.to(u.J)
    
    try:
        r_vals, u_vals = solve_radial_equation(E_n2_l1, 1, (r_min, r_max))
        
        plt.figure(figsize=(10, 6))
        plt.plot(r_vals / a0, u_vals, 'b-', linewidth=2, label='u(r) = r·R(r)')
        plt.xlabel('r / a₀')
        plt.ylabel('u(r)')
        plt.title('Wavefunction u(r) for l=1 at theoretical n=2 energy (-3.4 eV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        print(f"  Value at r_max: u({r_max/a0:.1f} a₀) = {u_vals[-1]:.6e}")
        print(f"  Should be close to zero for eigenvalue")
        
        plt.show()
        
    except Exception as e:
        print(f"  Failed to compute wavefunction: {e}")

if __name__ == "__main__":
    debug_l1_modes()
