"""
Advection in 2D
"""

#Create a simple advection solver in 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from godunov import advect
plt.ion()


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
c1 = ax1.contourf(X, Y, rho0, levels=50, cmap='viridis')
fig1.colorbar(c1)
ax1.set_title('Initial Condition')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Time-stepping parameters
dt = 0.02
nt = 800

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

#ani = FuncAnimation(fig2, update, frames=nt//10, blit=False, interval=50, repeat=False)
#plt.show()
# To save the animation, uncomment the following line
# ani.save('advection2D.mp4', writer='ffmpeg', fps=30