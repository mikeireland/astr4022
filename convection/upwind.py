"""
Upwind advection schemes for fluid dynamics. 

Normalised units:
g = 1, Hp = 1 = kT_eff/mu g
"""

import numpy as np

def advect(rho_in, u_in, v_in, dx, dy, dt, v_boundary="periodic"):
    """
    Perform one time step of advection using an upwind corner transport scheme.
    
    Parameters
    ----------
    rho_in : 2D array
        Scalar field to be advected.
    u : 2D array
        Velocity field in the x-direction.
    v : 2D array
        Velocity field in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dt : float
        Time step size.
        
    Returns
    -------
    rho : 2D array
        Updated scalar field after one time step.
    """
    # Expand rho, u and v to include ghost cells for periodic boundaries
    rho = np.pad(rho_in, pad_width=1, mode='wrap')
    u = np.pad(u_in, pad_width=1, mode='wrap')
    v = np.pad(v_in, pad_width=1, mode='wrap')

    # Apply periodic boundary conditions
    rho[0, :] = rho[-2, :]  # Bottom ghost row
    rho[-1, :] = rho[1, :]  # Top ghost row
    rho[:, 0] = rho[:, -2]  # Left ghost column
    rho[:, -1] = rho[:, 1]  # Right ghost column
    u[0, :] = u[-2, :]  # Bottom ghost row
    u[-1, :] = u[1, :]  # Top ghost row
    u[:, 0] = u[:, -2]  # Left ghost column
    u[:, -1] = u[:, 1]  # Right ghost column
    v[0, :] = v[-2, :]  # Bottom ghost row
    v[-1, :] = v[1, :]  # Top ghost row
    v[:, 0] = v[:, -2]  # Left ghost column
    v[:, -1] = v[:, 1]  # Right ghost column

    # If v_boundary is "continuous", apply continuous boundary conditions in y
    if v_boundary == "continuous":
        rho[0, :] = rho[1, :]  
        rho[-1, :] = rho[-2, :]  
    elif v_boundary == "reflective":
        rho[0,:] = 0
        rho[-1,:] = 0
    
    if v_boundary != "periodic":
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

    # Define the cells on the upwind side for the advection scheme.
    # Wherever u>0, we take the value from the left cell, else from the right cell.
    Cu = np.abs(u) * dt/dx
    Cv = np.abs(v) * dt/dy

    # Update the density field using the upwind values, using a periodic boundary condition
    # Here we use array slicing to avoid explicit loops for efficiency
    # For starters, ignore boundary cells
    # Do x-direction first
    rho[:, 1:-1]    = rho[:, 1:-1]   *  (1 - Cu[:, 1:-1]) + \
        Cu[:, 1:-1] * rho[:, :-2] * (u[:, 1:-1] > 0) + \
        Cu[:, 1:-1] * rho[:, 2:]  * (u[:, 1:-1] <= 0)
    # Then do y-direction
    rho[1:-1, 1:-1] = rho[1:-1, 1:-1] * (1 - Cv[1:-1, 1:-1]) + \
        Cv[1:-1, 1:-1] * rho[:-2, 1:-1] * (v[1:-1, 1:-1] > 0) + \
        Cv[1:-1, 1:-1] * rho[2:, 1:-1] * (v[1:-1, 1:-1] <= 0)
    
    # On the case of a "reflective" boundary at the top and bottom (y-direction),
    # we need to identify material that would advect out of the domain and reflect it back in.
    if v_boundary == "reflective":
        # Bottom boundary (y=0)
        outflow_bottom = (v[1, 1:-1] < 0) * Cv[1, 1:-1] * rho[1, 1:-1]
        rho[1, 1:-1] += outflow_bottom
        # Top boundary (y=Ny-1)
        outflow_top = (v[-2, 1:-1] > 0) * Cv[-2, 1:-1] * rho[-2, 1:-1]
        rho[-2, 1:-1] += outflow_top
    
    # Return the updated field without ghost cells
    rho = rho[1:-1, 1:-1]

    return rho

def accel(rho_in, p_in, u, v, dx, dy, dt, g=1.0):
    """
    Placeholder function for momentum equation solver.
    This will be implemented to handle acceleration/momentum updates.
    """

    # Compute pressure gradient
    dpdx = np.gradient(p_in, dx, axis=1)
    dpdy = np.gradient(p_in, dy, axis=0)

    # Update velocities based on pressure gradient
    u -= (dpdx / rho_in) * dt
    v -= (dpdy / rho_in) * dt
    
    # Update v due to gravity
    v -= g * dt

    # Advect the momentum fields
    rho_u = advect(u*rho_in, u, v, dx, dy, dt, v_boundary="continuous")
    rho_v = advect(v*rho_in, u, v, dx, dy, dt, v_boundary="continuous")

    return rho_u/rho_in, rho_v/rho_in