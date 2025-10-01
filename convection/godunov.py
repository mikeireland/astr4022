"""
Godunov advection schemes for fluid dynamics. 

We upwind the fluxes at cell boundaries and
include corner transport.

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
    # fluxes at cell boundaries and corners
    fluxes = np.zeros((rho_in.shape[0]+1, rho_in.shape[1]+1, 4))  
    
    # Lets start by ignoring corner transport
    # and just do 1D upwinding in x and y directions
    # For the interior cells, we compute the fluxes as a conditional sum
    fluxes[1:-1, :-1, 0] = 0.5*dt/dy * (\
        (u_in[:-1, :] * (u_in[:-1, :] > 0) * rho_in[:-1, :] +
         u_in[1:, :] * (u_in[1:, :] < 0) * rho_in[1:, :]))
    fluxes[:-1, 1:-1, 1] = 0.5*dt/dx * (\
        (v_in[:, :-1] * (v_in[:, :-1] > 0) * rho_in[:, :-1] +
         v_in[:, 1:] * (v_in[:, 1:] < 0) * rho_in[:, 1:]))
    # If we have periodic boundaries, we need to fill in the edges
    if v_boundary == "periodic":
        fluxes[0, :-1, 0] = 0.5*dt/dy * (\
            (u_in[-1, :] * (u_in[-1, :] > 0) * rho_in[-1, :] +
             u_in[0, :] * (u_in[0, :] < 0) * rho_in[0, :]))
        fluxes[-1, :-1, 0] = fluxes[0, :-1, 0]
        fluxes[:-1, 0, 1] = 0.5*dt/dx * (\
            (v_in[:, -1] * (v_in[:, -1] > 0) * rho_in[:, -1] +
             v_in[:, 0] * (v_in[:, 0] < 0) * rho_in[:, 0]))
        fluxes[:-1, -1, 1] = fluxes[:-1, 0, 1]
    # Now we can update the density field using the fluxes
    rho = rho_in - (fluxes[1:, :-1, 0] - fluxes[:-1, :-1, 0]) \
                   - (fluxes[:-1, 1:, 1] - fluxes[:-1, :-1, 1])
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