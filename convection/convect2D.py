"""
Convection in 2D. This is not driven convection, but simply
has a polytropic equation of state.

It totally fails if the initial conditions are such that 
we end up with supersonic velocities and shocks.
"""

# Create a simple convection solver in 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from godunov import advect, accel
plt.ion()

# Start with a simple uniform density, a polytropic equation
# of state, and a small perturbation in the center of the domain
nx, ny = 100, 100
Lx, Ly = 10.0, 10.0
dx, dy = Lx/nx, Ly/ny
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
rho0 = 4*np.exp(-0.35*Y)#np.ones((ny, nx))
rho0[40:50, 30:40] = 4  
rho0[20:30, 70:80] = 0.5  
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
nt = 4000

# Set up figure 2 for the animation
fig2, ax2 = plt.subplots()
c2 = ax2.contourf(X, Y, rho0, levels=50, cmap='viridis', vmax=3.0, vmin=0.1)
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
    # If we don't do this, there is no initial plot...
    if (frame == 0):
        c2 = ax2.contourf(X, Y, rho, levels=50, cmap='viridis', vmax=3.0, vmin=0.1)
        return c2.collections
    for _ in range(10):  # Take 10 time steps per frame for faster animation
        rho = advect(rho, u, v, dx, dy, dt, v_boundary="reflective")
        p = rho**1.4  # Update pressure with polytropic EOS
        u, v = accel(rho, p, u, v, dx, dy, dt)
    ax2.clear()
    c2 = ax2.contourf(X, Y, rho, levels=50, cmap='viridis', vmax=3.0, vmin=0.1)
    ax2.set_title('Convection in 2D')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y') 
    return c2.collections

data = [rho, p, u, v]
def reset(data):
    data[0] = rho0.copy()
    data[1] = p0.copy()
    data[2] = u0.copy()
    data[3] = v0.copy()
    
def run(n, data):
    rho, p, u, v = data
    this_dt = dt
    for ix in range(n):  # Take n time steps
        max_dt = 0.5 * min(dx/np.max(np.abs(u)+1e-10), dy/np.max(np.abs(v)+1e-10))
        if this_dt > max_dt:
            print(f"Adjusting dt from {this_dt} to {max_dt} for stability at step {ix}")
            this_dt = max_dt
        rho = advect(rho, u, v, dx, dy, this_dt, v_boundary="reflective")
        #Ensure non-negative density
        if np.min(rho) < 0:
            print("Negative density detected, blurring and stopping (please check!)")
            rho = 0.25*(np.roll(rho, 1, axis=1) + np.roll(rho, -1, axis=1)) + 0.5*rho
            import pdb; pdb.set_trace()
        p = rho**1.4  # Update pressure with polytropic EOS
        u, v = accel(rho, p, u, v, dx, dy, this_dt)
        # This should solve for a supersonic velocity.
        if np.max(np.abs(u)) > 10.0 or np.max(np.abs(v)) > 10.0:
            print("Velocity too large, stopping (please check!)")
            import pdb; pdb.set_trace()
    return [rho, p, u, v]

ani = FuncAnimation(fig2, update, frames=nt//10, blit=False, interval=100, repeat=False)
plt.show()
# Save the animation, by uncomment the following line instead of plt.show above.
#ani.save('convection2D.mp4', writer='ffmpeg', fps=30)