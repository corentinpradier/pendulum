import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Définition des constantes 
l1, l2, l3 = sp.symbols('l1 l2 l3', positive=True)
m1, m2, m3 = sp.symbols('m1 m2 m3', positive=True)
g = sp.symbols('g', positive=True)
t = sp.symbols('t')

# Définition des angles comme fonctions du temps 
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta3 = sp.Function('theta3')(t)


x1 = l1 * sp.sin(theta1)
y1 = -l1 * sp.cos(theta1)

x2 = x1 + l2 * sp.sin(theta2)
y2 = y1 - l2 * sp.cos(theta2)

x3 = x2 + l3 * sp.sin(theta3)
y3 = y2 - l3 * sp.cos(theta3)

dx1 = sp.diff(x1, t)
dy1 = sp.diff(y1, t)
dx2 = sp.diff(x2, t)
dy2 = sp.diff(y2, t)
dx3 = sp.diff(x3, t)
dy3 = sp.diff(y3, t)

# Energie cinétique
T = sp.Rational(1, 2) * m1 * (dx1**2 + dy1**2) + \
    sp.Rational(1, 2) * m2 * (dx2**2 + dy2**2) + \
    sp.Rational(1, 2) * m3 * (dx3**2 + dy3**2)

T = sp.simplify(T)

# Energie potentielle
V = g * (m1 * y1 + m2 * y2 + m3 * y3)

# Lagrangien
L = sp.simplify(T - V)

# Euler-Lagrange
omega1 = sp.diff(theta1, t)
omega2 = sp.diff(theta2, t)
omega3 = sp.diff(theta3, t)

def euler_lagrange(L, theta_i, omega_i):
    dL_omega = sp.diff(L, omega_i)
    d_dt_dL_omega = sp.diff(dL_omega, t)
    dL_theta = sp.diff(L, theta_i)
    return sp.simplify(d_dt_dL_omega - dL_theta)

eq1 = euler_lagrange(L, theta1, omega1)
eq2 = euler_lagrange(L, theta2, omega2)
eq3 = euler_lagrange(L, theta3, omega3)

# Résolution des équations
theta1_dd = sp.diff(theta1, t, 2)
theta2_dd = sp.diff(theta2, t, 2)
theta3_dd = sp.diff(theta3, t, 2)

M, b = sp.linear_eq_to_matrix([eq1, eq2, eq3], [theta1_dd, theta2_dd, theta3_dd])

variables = (theta1, theta2, theta3, omega1, omega2, omega3, m1, m2, m3, l1, l2, l3, g)
M_func = sp.lambdify(variables, M, modules='numpy')
b_func = sp.lambdify(variables, b, modules='numpy')


def pendulum_to_solve(t, U, m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val):
    th1, om1, th2, om2, th3, om3 = U

    M_num = M_func(th1, th2, th3, om1, om2, om3, m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val)
    b_num = b_func(th1, th2, th3, om1, om2, om3, m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val)

    theta_dd = np.linalg.solve(M_num, b_num).flatten()  

    return [om1, theta_dd[0], om2, theta_dd[1], om3, theta_dd[2]]


def pendulum_solver(f, U_0, t_eval, args):
    m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val = args
    result = solve_ivp(f, (0, 100), U_0, t_eval=t_eval, args=args)

    # On repasse en coordonnées cartésiennes pour le tracé
    theta1_val = result.y[0]
    theta2_val = result.y[2]
    theta3_val = result.y[4]

    x1_val = l1_val * np.sin(theta1_val)
    y1_val = -l1_val * np.cos(theta1_val)

    x2_val = x1_val + l2_val * np.sin(theta2_val)
    y2_val = y1_val - l2_val * np.cos(theta2_val)

    x3_val = x2_val + l3_val * np.sin(theta3_val)
    y3_val = y2_val - l3_val * np.cos(theta3_val)

    return [x1_val, y1_val, x2_val, y2_val, x3_val, y3_val]


def show_animation(fig, ax, X, t_eval, lengths, color='red'):
    l1_val, l2_val, l3_val = lengths
    length_sum = l1_val + l2_val + l3_val
    x1, y1, x2, y2, x3, y3 = X

    ax.set_xlim(-(length_sum)*1.1, (length_sum)*1.1)
    ax.set_ylim(-(length_sum)*1.1, (length_sum)*1.1)
    ax.set_aspect("equal")

    # éléments à mettre à jour : les deux tiges + les deux masses + la trace
    line, = ax.plot([], [], 'o-', lw=2, color='black')       # tiges + masses
    trace, = ax.plot([], [], '-', lw=1, color=color, alpha=0.5)  # trace de m2

    trace_x, trace_y = [], []

    def update(frame):
        # positions à cet instant
        thisx = [0, x1[frame], x2[frame], x3[frame]]
        thisy = [0, y1[frame], y2[frame], y3[frame]]
        line.set_data(thisx, thisy)

        trace_x.append(x3[frame])
        trace_y.append(y3[frame])
        trace.set_data(trace_x, trace_y)

        return line, trace

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=1000/20, blit=False)
    return ani


def main():
    l1_val, l2_val, l3_val = 1, 1, 1
    m1_val, m2_val, m3_val = 1, 10, 1
    g_val = 9.8
    epsilon = 1e-5

    n_step = np.linspace(0, 100, 2000)
    U_0 = [np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]
    U_1 = [np.pi/2, 0, np.pi/2+epsilon, 0, np.pi/2, 0]

    res1 = pendulum_solver(pendulum_to_solve, U_0, n_step, args=(m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val))
    res2 = pendulum_solver(pendulum_to_solve, U_1, n_step, args=(m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, g_val))

    fig, ax = plt.subplots()
    ani1 = show_animation(fig, ax, res1, n_step, [l1_val, l2_val, l3_val])
    # ani2 = show_animation(fig, ax, res2, n_step, [l1_val, l2_val, l3_val], color='blue')
    plt.show()

main()