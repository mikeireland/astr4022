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
    Perform one time step of advection using an Godonov corner transport scheme.
    
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
    
    Cu = dt/dx * np.abs(u_in)
    Cv = dt/dy * np.abs(v_in)
    # Lets start with 1D upwinding in x and y directions, 
    # then we will add corner transport terms
    # For the interior cells, we compute the fluxes as a conditional sum
    # Coordinates are y then x.
    fluxes[1:-1, :-1, 0] = \
        (1-Cu[:-1, :])*Cv[:-1, :] * (v_in[:-1, :] > 0) * rho_in[:-1, :] - \
        (1-Cu[1:, :])*Cv[1:, :] * (v_in[1:, :] < 0) * rho_in[1:, :]
    fluxes[:-1, 1:-1, 1] = \
        (1-Cv[:, :-1])*Cu[:, :-1] * (u_in[:, :-1] > 0) * rho_in[:, :-1] - \
        (1-Cv[:, 1:])*Cu[:, 1:] * (u_in[:, 1:] < 0) * rho_in[:, 1:]
    # Now add corner transport terms
    fluxes[1:-1, 1:-1, 2] = \
        Cu[:-1, :-1] * Cv[:-1, :-1] * (u_in[:-1, :-1] > 0) * (v_in[:-1, :-1] > 0) * rho_in[:-1, :-1] - \
        Cu[1:,1:] * Cv[1:,1:] * (u_in[1:,1:] < 0) * (v_in[1:,1:] < 0) * rho_in[1:,1:]
    fluxes[1:-1, 1:-1, 3] = \
        Cu[1:, :-1] * Cv[1:, :-1] * (u_in[1:, :-1] < 0) * (v_in[1:, :-1] > 0) * rho_in[1:, :-1] - \
        Cu[:-1, 1:] * Cv[:-1, 1:] * (u_in[:-1, 1:] > 0) * (v_in[:-1, 1:] < 0) * rho_in[:-1, 1:]
    # Combine corner transport terms into x and y fluxes
    # If we have periodic boundaries, we need to fill in the edges
    # x boundary is always periodic.
    # Horizontal fluxes in left and right column
    fluxes[:-1, 0, 1] = \
        (1 - Cv[:, -1])*Cu[:, -1] * (u_in[:, -1] > 0) * rho_in[:, -1] - \
        (1 - Cv[:, 0])*Cu[:, 0] * (u_in[:, 0] < 0) * rho_in[:, 0]
    fluxes[:-1, -1, 1] = fluxes[:-1, 0, 1]
    # (0,0) Corner transport terms in left and right column
    #fluxes[:-1, 0, 2] = \
    #    Cu[:, -1] * Cv[:,-1] * (u_in[:, -1] > 0) * (v_in[:,-1] > 0) * rho_in[:, -1] - \
    #    Cu[:, 0] * Cv[:,0] * (u_in[:, 0] < 0) * (v_in[:,0] < 0) * rho_in[:, 0]
    #fluxes[:-1, -1, 2] = fluxes[:-1, 0, 2]
    # (0,1) Corner transport terms in left and right column
    #fluxes[:-1, 0, 3] = \
    #    Cu[:, -1] * Cv[-1, :] * (u_in[:, -1] < 0) * (v_in[-1, :] > 0) * rho_in[:, -1] - \
    #    Cu[:, 0] * Cv[0, :] * (u_in[:, 0] > 0) * (v_in[0, :] < 0) * rho_in[:, 0]
    #fluxes[:-1, -1, 3] = fluxes[:-1, 0, 3]
    if v_boundary == "periodic":
        # Vertical fluxes in top bottom row and top row. 
        fluxes[0, :-1, 0] = \
            (1 - Cu[-1,:])*Cv[-1, :] * (v_in[-1, :] > 0) * rho_in[-1, :] - \
            (1 - Cu[0,:])*Cv[0, :] * (v_in[0, :] < 0) * rho_in[0, :]
        fluxes[-1, :-1, 0] = fluxes[0, :-1, 0]
        # (0,0) Corner transport terms in top and bottom row. 
        #fluxes[0, :-1, 2] = \
        #    Cu[-1, :] * Cv[-1, :] * (u_in[-1, :] > 0) * (v_in[-1, :] > 0) * rho_in[-1, :] - \
        #    Cu[0, :] * Cv[0, :] * (u_in[0, :] < 0) * (v_in[0, :] < 0) * rho_in[0, :]
        #fluxes[-1, :, 2] = fluxes[0, :, 2]
        
    # Now we can update the density field using the fluxes
    rho = rho_in - (fluxes[1:, :-1, 0] - fluxes[:-1, :-1, 0]) \
                 - (fluxes[:-1, 1:, 1] - fluxes[:-1, :-1, 1]) \
                 - (fluxes[1:, 1:, 2] - fluxes[:-1, :-1, 2]) \
                 - (fluxes[1:, :-1, 3] - fluxes[:-1, 1:, 3])

    return rho

def accel(rho_in, p_in, u, v, dx, dy, dt, g=1.0):
    """
    Function for momentum equation solver.
    
    Importantly, a pressure gradient can not modify velocity independently to 
    density. For highly subsonic flows, we can consider the boundary between cells 
    like two connected cylinders. 
    F = A*dP
    m_move = 0.5*V(rho_1 + rho_2)
    a = F/m = 2 * dP / (rho_1 + rho_2) / dx
    s = 0.5*a*dt**2
    drho = dm/V = 0.5*a*dt**2 / dx
         = 2 * dP / (rho_1 + rho_2) * (dt/dx)**2
    dp = v (rho_1 - drho)
       = 2 * dP / (rho_1 + rho_2) * (dt / dx) * \
           (rho_1 - drho)
           
    With our example: 
    - rho_1=2, rho_2=1, p_1=2.6, p_2=1, (dt/dx)=0.1
    - drho = 0.0053
    - dp = 0.32
    - v_2 = 0.32 and v_1 = 0.16. 
    ... do we need to add energy?
    """

    if False:
        # Simple minded approach: just update velocities based on pressure gradient
        dpdx = np.gradient(p_in, dx, axis=1)
        dpdy = np.gradient(p_in, dy, axis=0)

        # Update velocities based on pressure gradient
        u -= (dpdx / rho_in) * dt
        v -= (dpdy / rho_in) * dt

    # Advect the momentum fields
    rho_u = advect(u*rho_in, u, v, dx, dy, dt, v_boundary="reflective")
    rho_v = advect(v*rho_in, u, v, dx, dy, dt, v_boundary="reflective")
    u = rho_u/rho_in
    v = rho_v/rho_in

    # Compute x-direction velocity change due to pressure gradient,
    # at cell boundaries. This is periodic.
    dv_x_face = np.zeros((u.shape[0], u.shape[1]+1))
    dv_x_face[:, 1:-1] = 2 * (p_in[:, 1:] - p_in[:, :-1]) * dt/dx / \
         (rho_in[:, 1:] + rho_in[:, :-1]) 
    dv_x_face[:, -1] = 2 * (p_in[:, 0] - p_in[:, -1]) * dt/dx / \
         (rho_in[:, 0] + rho_in[:, -1])
    dv_x_face[:, 0] = dv_x_face[:, -1]
    
    # Compute y-direction velocity change due to pressure gradient,
    # at cell boundaries. The top and bottom is a copy of the adjacent cell.
    dv_y_face = np.zeros((v.shape[0]+1, v.shape[1]))
    dv_y_face[1:-1, :] = 2 * (p_in[1:, :] - p_in[:-1, :]) * dt/dy / \
         (rho_in[1:, :] + rho_in[:-1, :])
    dv_y_face[0, :] = dv_y_face[1, :]  # Reflective boundary at bottom
    dv_y_face[-1, :] = dv_y_face[-2, :]  # Reflective boundary at top

    #NB!!! We should also change the density field here to account for
    # the mass that moves with the velocity change. But we will ignore this
    # for now and "fix" with a bit of viscosity.

    # Update velocities at cell centers by summing face values
    u -= (dv_x_face[:, 1:] + dv_x_face[:, :-1])
    v -= (dv_y_face[1:, :] + dv_y_face[:-1, :])

    # Update v due to gravity
    v -= g * dt
    
    # Add a small amount of viscosity to stabilize things. 
    # !!! This is a hack.
    u = 0.3*(np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + \
              np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1))/4 + 0.7*u
    v = 0.3*(np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + \
              np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1))/4 + 0.7*v


    return u,v