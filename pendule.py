import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

class Pendulum:
    g = 9.8

    def __init__(self, order, length, mass):
        self.order = order
        self.length = length
        self.mass = mass
        self.theta_indices = range(0, 2*self.order, 2)

        if self.order == 1:
            self.pendulum_to_solve = self._simple_pendulum
        if self.order == 2:
            self.pendulum_to_solve = self._double_pendulum
        if self.order == 3:
            self.M_func, self.b_func = self._calc_triple_pendulum()
            self.pendulum_to_solve = self._triple_pendulum

    def _simple_pendulum(self, t, U):
        theta, omega = U
        r = self.length

        return [omega, -self.g/r * np.sin(theta)]


    def _double_pendulum(self, t, U):
        theta1, omega1, theta2, omega2 = U
        dtheta = theta1 - theta2
        s, c = np.sin(dtheta), np.cos(dtheta)
        l1, l2 = self.length
        m1, m2 = self.mass

        delta = l1 * l2 * (m1 + m2 * s**2)
        d_omega1 = (l2 * (-m2 * l2 * s * omega2**2 - (m1 + m2) * self.g * np.sin(theta1))
                    - m2 * l2 * c * (l1 * s * omega1**2 - self.g * np.sin(theta2)))
        d_omega2 = ((m1 + m2) * l1 * (l1 * s * omega1**2 - self.g * np.sin(theta2))
                    - l1 * c * (-m2 * l2 * s * omega2**2 - (m1 + m2) * self.g * np.sin(theta1)))

        return [omega1, d_omega1 / delta, omega2, d_omega2 / delta]


    def _calc_triple_pendulum(self):
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

        return M_func, b_func


    def _triple_pendulum(self, t, U):
        th1, om1, th2, om2, th3, om3 = U
        l1_val, l2_val, l3_val = self.length
        m1_val, m2_val, m3_val = self.mass

        M_num = self.M_func(th1, th2, th3, om1, om2, om3, m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, self.g)
        b_num = self.b_func(th1, th2, th3, om1, om2, om3, m1_val, m2_val, m3_val, l1_val, l2_val, l3_val, self.g)

        theta_dd = np.linalg.solve(M_num, b_num).flatten()  

        return [om1, theta_dd[0], om2, theta_dd[1], om3, theta_dd[2]]


    def pendulum_solver(self, U_0: list, start: int, end: int, n_step: int):
        t_eval = np.linspace(start, end, n_step)
        self.result = solve_ivp(self.pendulum_to_solve, (start, end), U_0, t_eval=t_eval)
    
        theta = np.array([self.result.y[i] for i in self.theta_indices])
        lengths = np.array(self.length).reshape(-1, 1)

        dx = lengths * np.sin(theta)  
        dy = -lengths * np.cos(theta)

        self.X = np.cumsum(dx, axis=0)
        self.Y = np.cumsum(dy, axis=0)


    def show_animation(self, fig=None, ax=None, interval=100, color='red', show=True):
        sum_tot = np.sum(self.length)

        if show:
            fig, ax = plt.subplots(figsize=(6,6))

        ax.set_xlim(-sum_tot*1.1, sum_tot*1.1)
        ax.set_ylim(-sum_tot*1.1, sum_tot*1.1)
        ax.set_aspect("equal")

        # éléments à mettre à jour : les deux tiges + les deux masses + la trace
        line, = ax.plot([], [], 'o-', lw=2, color='black')       # tiges + masses
        trace, = ax.plot([], [], '-', lw=1, color=color, alpha=0.5)  # trace de m2

        trace_x, trace_y = [], []

        def update(frame):
            # positions à cet instant
            thisx, thisy = [0], [0]
            thisx.extend(self.X[i][frame] for i in range(self.order))
            thisy.extend(self.Y[i][frame] for i in range(self.order))
            line.set_data(thisx, thisy)

            trace_x.append(self.X[-1][frame])
            trace_y.append(self.Y[-1][frame])
            trace.set_data(trace_x, trace_y)

            return line, trace

        ani = FuncAnimation(fig, update, frames=len(self.result.t), interval=interval, blit=False)
        
        if show:    
            plt.show()
        else:
            return ani