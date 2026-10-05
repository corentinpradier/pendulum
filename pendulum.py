import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

class Pendulum:
    g = 9.8

    def __init__(self, order, length, mass, damping=None, method='RK45'):
        """
            Args:
                order (int) : Nombre de tiges du pendule.
                length (list) : Liste des longueurs des tiges.
                mass (list) : Liste des masses.
                damping (list) : Liste des coefficients d'amortissement. Défaut : None.
                method (str) : Méthode d'intégration. Défaut : 'RK45'.
        """
        if not hasattr(length, '__iter__'):
            length = [length]
        if not hasattr(mass, '__iter__'):
            mass = [mass]

        self.order = order
        self.length = length
        self.mass = mass
        self.damping = damping if damping is not None else [0]*order
        self.method = method
        self.theta_indices = range(0, 2*self.order, 2)

        self.M_func, self.b_func, self.T_func, self.V_func = self._calc_general_pendulum()
        self.pendulum_to_solve = self._general_pendulum


    def _calc_general_pendulum(self):
        # Définition des constantes 
        li = [sp.symbols(f'l{i}', positive=True) for i in range(self.order)]
        mi = [sp.symbols(f'm{i}', positive=True) for i in range(self.order)]
        ci = [sp.symbols(f'c{i}', nonnegative=True) for i in range(self.order)]  # ← nouveau
        g = sp.symbols('g', positive=True)
        t = sp.symbols('t')

        # Définition des angles comme fonctions du temps 
        thetai = [sp.Function(f'theta{i}')(t) for i in range(self.order)]

        xi, yi = [], []
        dxi, dyi = [], []
        x_cum, y_cum = 0, 0
        for i in range(self.order):
            x_cum += li[i] * sp.sin(thetai[i])
            y_cum +=  -li[i] * sp.cos(thetai[i])

            xi.append(x_cum)
            yi.append(y_cum)

            dxi.append(sp.diff(x_cum, t))
            dyi.append(sp.diff(y_cum, t))

        # Energie cinétique et potentielle
        T, V = 0, 0
        for k in range(self.order):
            T += sp.Rational(1, 2) * mi[k] * (dxi[k]**2 + dyi[k]**2)
            V += g * mi[k] * yi[k]

        T = sp.simplify(T)

        # Lagrangien
        L = sp.simplify(T - V)

        # Euler-Lagrange
        omegai = [sp.diff(thetai[i], t) for i in range(self.order)]

        def euler_lagrange(L, theta_i, omega_i):
            dL_omega = sp.diff(L, omega_i)
            d_dt_dL_omega = sp.diff(dL_omega, t)
            dL_theta = sp.diff(L, theta_i)
            return sp.simplify(d_dt_dL_omega - dL_theta)

        eqi = [euler_lagrange(L, thetai[k], omegai[k]) + ci[k] * omegai[k] 
               for k in range(self.order)]

        # Résolution des équations
        thetai_dd = [sp.diff(thetai[k], t, 2) for k in range(self.order)]

        M, b = sp.linear_eq_to_matrix(eqi, thetai_dd)

        variables = tuple(thetai + omegai + mi + li + ci + [g])
        energy_var = tuple(thetai + omegai + mi + li + [g])

        M_func = sp.lambdify(variables, M, modules='numpy')
        b_func = sp.lambdify(variables, b, modules='numpy')

        T_func = sp.lambdify(energy_var, T, modules='numpy')
        V_func = sp.lambdify(energy_var, V, modules='numpy')

        return M_func, b_func, T_func, V_func


    def _general_pendulum(self, t, U):
        thetas = U[0::2]   # tous les angles : indices 0, 2, 4, ...
        omegas = U[1::2]   # toutes les vitesses angulaires : indices 1, 3, 5, ...

        args = list(thetas) + list(omegas) + list(self.mass) + \
               list(self.length) + list(self.damping) + [self.g]

        M_num = np.array(self.M_func(*args), dtype=float)
        b_num = np.array(self.b_func(*args), dtype=float)

        theta_dd = np.linalg.solve(M_num, b_num).flatten()

        # reconstruit [omega1, theta_dd1, omega2, theta_dd2, ...]
        dU = []
        for k in range(self.order):
            dU.append(omegas[k])
            dU.append(theta_dd[k])
        return dU


    def pendulum_solver(self, U_0: list, start: int, end: int, n_step: int):
        t_eval = np.linspace(start, end, n_step)
        self.result = solve_ivp(
            self.pendulum_to_solve, (start, end), U_0,
            t_eval=t_eval,
            method=self.method,
            rtol=1e-10,
            atol=1e-10    
        )

        theta = np.array([self.result.y[i] for i in self.theta_indices])
        lengths = np.array(self.length).reshape(-1, 1)

        dx = lengths * np.sin(theta)  
        dy = -lengths * np.cos(theta)

        self.X = np.cumsum(dx, axis=0)
        self.Y = np.cumsum(dy, axis=0)

        self.T_values, self.V_values = self._compute_energy()
        self.E_values = self.T_values + self.V_values

    
    def _compute_energy(self):
        thetas = self.result.y[0::2]
        omegas = self.result.y[1::2]

        n_step = thetas.shape[1]
        T_arr = np.zeros(n_step)
        V_arr = np.zeros(n_step)

        for k in range(n_step):
            args = (list(thetas[:, k]) + list(omegas[:, k])
                + list(self.mass) + list(self.length) + [self.g])

            T_arr[k] = self.T_func(*args)
            V_arr[k] = self.V_func(*args)

        return T_arr, V_arr


    def show_animation(self, fig=None, ax=None, interval=100, color='red', show=True, title=None):
        sum_tot = np.sum(self.length)

        if show:
            fig, ax = plt.subplots(figsize=(6,6))
        if title is not None:
            ax.set_title(title)

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