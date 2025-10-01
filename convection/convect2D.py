"""
Convection in 2D. This is not driven convection, but simply
has a polytropic equation of state.
"""

# Create a simple convection solver in 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from upwind import advect, accel
plt.ion()

# Start with a simple uniform density, a polytropic equation
# of state, and a small perturbation in the center of the domain
nx, ny = 100, 100
Lx, Ly = 10.0, 10.0
dx, dy = Lx/nx, Ly/ny
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
rho0 = np.ones((ny, nx))
rho0[45:55, 45:55] = 2.0  
p0 = rho0**1.4  # Polytropic EOS, gamma=1.4
u0 = np.zeros_like(rho0)  # Initial velocity in x-direction
v0 = np.zeros_like(rho0)  # Initial velocity in y-direction

# In figure 1, we will plot the initial condition
fig1, ax1 = plt.subplots()
c1 = ax1.contourf(X, Y, rho0, levels=50, cmap='viridis')
fig1.colorbar(c1)
ax1.set_title('Initial Condition')
ax1.set_xlabel('x')
ax1.set_ylabel('y') 

# Time-stepping parameters
dt = 0.01
nt = 500

# Set up figure 2 for the animation
fig2, ax2 = plt.subplots()
c2 = ax2.contourf(X, Y, rho0, levels=50, cmap='viridis')
fig2.colorbar(c2)
ax2.set_title('Convection in 2D')
ax2.set_xlabel('x')
ax2.set_ylabel('y') 

rho = rho0.copy()
p = p0.copy()
u = u0.copy()
v = v0.copy()
# Animation update function
def update(frame):
    global rho, p, u, v
    for _ in range(10):  # Take 10 time steps per frame for faster animation
        rho = advect(rho, u, v, dx, dy, dt, v_boundary="reflective")
        p = rho**1.4  # Update pressure with polytropic EOS
        u, v = accel(rho, p, u, v, dx, dy, dt)
    ax2.clear()
    c2 = ax2.contourf(X, Y, rho, levels=50, cmap='viridis')
    ax2.set_title('Convection in 2D')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y') 
    return c2.collections

ani = FuncAnimation(fig2, update, frames=nt//10, blit=False, interval=50, repeat=False)
plt.show()