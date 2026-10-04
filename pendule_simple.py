import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

r = 1
g = 9.8
start, end = 0, 100
step = 1000

def f(t, U):
    theta, omega = U
    return [omega, -g/r * np.sin(theta)]

U_0 = [np.pi/2, 0]
t_eval = np.linspace(start, end, step)
result = solve_ivp(f, (start, end), U_0, t_eval=t_eval)

theta = result.y[0]

x = r * np.sin(theta)
y = -r * np.cos(theta)

fig, ax = plt.subplots(figsize=(15,6))
ax.set_xlim(-r*1.1, r*1.1)
ax.set_ylim(-r*1.1, r*1.1)
ax.set_aspect("equal")

line, = ax.plot([], [], 'o-', lw=2, color='black')
trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)

trace_x, trace_y = [], []

def update(frame):
    thisx = [0, x[frame]]
    thisy = [0, y[frame]]
    line.set_data(thisx, thisy)

    trace_x.append(x[frame])
    trace_y.append(y[frame])
    trace.set_data(trace_x, trace_y)

    return line, trace

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=100, blit=True)
plt.show()