import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 100.0  # Length of the domain
N = 512  # Number of spatial points
dx = L / N  # Spatial step size
dt = 0.01  # Time step size
T = 10.0  # Total time
c = 1.0  # Speed of the soliton

# Domain of graph
x = np.linspace(0, L, N)

# Initial condition
uInitial = 0.5 * c * np.cosh(0.5 * np.sqrt(c) * (x - L / 2))**-2
u = uInitial.copy()

# KdV function
def kdv_rhs(u):
    u_x = np.gradient(u, dx)
    u_xxx = np.gradient(np.gradient(np.gradient(u, dx), dx), dx)
    return -6 * u * u_x - u_xxx

# Some Runge-Jutta method I found on the internet used for time evolution
# WIll prob edit later
def rk4_step(u, dt):
    k1 = dt * kdv_rhs(u)
    k2 = dt * kdv_rhs(u + 0.5 * k1)
    k3 = dt * kdv_rhs(u + 0.5 * k2)
    k4 = dt * kdv_rhs(u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Plot
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y (u)')
ax.set_title('Soliton Evolution through KdV differential equation')

# Animation
def animate(frame):
    global u
    for _ in range(10):  # Update the solution multiple times per frame
        u = rk4_step(u, dt)
    line.set_ydata(u)
    return line,

ani = FuncAnimation(fig, animate, frames=int(T / dt / 10), interval=50, blit=True)
plt.show()
