"""
Advection in 2D
"""

#Create a simple advection solver in 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Set up our grid and initial conditions, of 
# a Gaussian blob in the center of the domain
nx, ny = 100, 100
Lx, Ly = 10.0, 10.0
dx, dy = Lx/nx, Ly/ny
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
sigma = 0.5
rho0 = np.exp(-((X-Lx/2)**2 + (Y-Ly/2)**2)/(2*sigma**2))

# Make a square blob instead
rho0 = np.zeros((ny, nx))
rho0[40:60, 40:60] = 1.0

# Initial velocity field (constant rightward and upward)
u = -1.0  # velocity in x-direction
v = -0.5  # velocity in y-direction
u0 = np.ones_like(rho0) * u
v0 = np.ones_like(rho0) * v

# In figure 1, we will plot the initial condition
fig1, ax1 = plt.subplots()
c1 = ax1.contourf(X, Y, u0, levels=50, cmap='viridis')
fig1.colorbar(c1)
ax1.set_title('Initial Condition')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Time-stepping parameters
dt = 0.05
nt = 100

# Function to perform one time step of the advection.
# Use an upwind corner transport scheme for stability.
def advect(rho_in, u, v, dx, dy, dt):
    """
    Parameters
    ----------
    rho : 2D array
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
    """
    # Expand rho, u and v to include ghost cells for periodic boundaries
    rho = np.pad(rho_in, pad_width=1, mode='wrap')
    u = np.pad(u, pad_width=1, mode='wrap')
    v = np.pad(v, pad_width=1, mode='wrap')
    
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
        
    # Return the updated field without ghost cells
    rho = rho[1:-1, 1:-1]

    return rho

# Set up figure 2 for the animation
fig2, ax2 = plt.subplots()
c2 = ax2.contourf(X, Y, rho0, levels=50, cmap='viridis')
fig2.colorbar(c2)
ax2.set_title('Advection of Scalar Field')
ax2.set_xlabel('x')
ax2.set_ylabel('y') 

rho = rho0.copy()
# Animation update function
def update(frame):
    global rho
    for _ in range(10):  # Take 10 time steps per frame for faster animation
        rho = advect(rho, u0, v0, dx, dy, dt)
    ax2.clear()
    c2 = ax2.contourf(X, Y, rho, levels=50, cmap='viridis')
    ax2.set_title('Advection of Scalar Field')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y') 
    return c2.collections   

ani = FuncAnimation(fig2, update, frames=nt//10, blit=False, interval=100, repeat=False)
plt.show()
# To save the animation, uncomment the following line
# ani.save('advection2D.mp4', writer='ffmpeg', fps=30