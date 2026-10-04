import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# Parameters 
m1, l1 = 10, 1
m2, l2 = 1, 2
g = 10

def f(t, U):
    theta1, omega1, theta2, omega2 = U
    dtheta = theta1 - theta2
    s, c = np.sin(dtheta), np.cos(dtheta)

    delta = l1 * l2 * (m1 + m2 * s**2)
    d_omega1 = (l2 * (-m2 * l2 * s * omega2**2 - (m1 + m2) * g * np.sin(theta1))
                - m2 * l2 * c * (l1 * s * omega1**2 - g * np.sin(theta2)))
    d_omega2 = ((m1 + m2) * l1 * (l1 * s * omega1**2 - g * np.sin(theta2))
                - l1 * c * (-m2 * l2 * s * omega2**2 - (m1 + m2) * g * np.sin(theta1)))

    return [omega1, d_omega1 / delta, omega2, d_omega2 / delta]


def pendulum_solver(f, U_0, t_eval):
    # U_0 = [np.pi, 1, np.pi, 0] # CI du donut

    result = solve_ivp(f, (0, 100), U_0, t_eval=t_eval)

    # On repasse en coordonnées cartésiennes pour le tracé
    theta1 = result.y[0]
    theta2 = result.y[2]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return [x1, y1, x2, y2]


def show_animation(fig, ax, X, t_eval, color='red'):
    x1, y1, x2, y2 = X
    ax.set_xlim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax.set_ylim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax.set_aspect("equal")

    # éléments à mettre à jour : les deux tiges + les deux masses + la trace
    line, = ax.plot([], [], 'o-', lw=2, color='black')       # tiges + masses
    trace, = ax.plot([], [], '-', lw=1, color=color, alpha=0.5)  # trace de m2

    trace_x, trace_y = [], []

    def update(frame):
        # positions à cet instant
        thisx = [0, x1[frame], x2[frame]]
        thisy = [0, y1[frame], y2[frame]]
        line.set_data(thisx, thisy)

        trace_x.append(x2[frame])
        trace_y.append(y2[frame])
        trace.set_data(trace_x, trace_y)

        return line, trace

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=100, blit=False)
    return ani

def main():
    n_step = np.linspace(0, 100, 2000)

    U_0 = [np.pi/2, 0, np.pi/2, 0]
    U_1 = [np.pi/2 + 1e-5, 0, np.pi/2, 0]

    res1 = pendulum_solver(f, U_0, n_step)
    res2 = pendulum_solver(f, U_1, n_step)


    fig, ax = plt.subplots(figsize=(6,6))
    ani1 = show_animation(fig, ax, res1, n_step)
    ani2 = show_animation(fig, ax, res2, n_step, color='blue')
    plt.show()

main()