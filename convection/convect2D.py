"""
Convection in 2D. This is not driven convection, but simply
has a polytropic equation of state.
"""

#Create a simple convection solver in 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from upwind import advect
plt.ion()

