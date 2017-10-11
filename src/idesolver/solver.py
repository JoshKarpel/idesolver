import warnings
import logging

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConvergenceWarning(Warning):
    pass


def complex_quadrature(integrand, a, b, **kwargs):
    def real_func(x):
        return np.real(integrand(x))

    def imag_func(x):
        return np.imag(integrand(x))

    real_integral = integ.quadrature(real_func, a, b, **kwargs)
    imag_integral = integ.quadrature(imag_func, a, b, **kwargs)

    return real_integral[0] + 1j * imag_integral[0], real_integral[1], imag_integral[1]


class IDESolver:
    dtype = np.float64
    ode_solver = integ.ode

    def __init__(self,
                 y_initial,
                 x,
                 c = None,
                 d = None,
                 k = None,
                 f = None,
                 lower_bound = None,
                 upper_bound = None,
                 global_error_tolerance = 1e-9,
                 interpolation_kind = 'cubic',
                 max_iterations = None,
                 smoothing_factor = .5):
        self.y_initial = y_initial
        self.x = np.array(x)

        if c is None:
            c = lambda x, y: 0
        if d is None:
            d = lambda x: 0
        if k is None:
            k = lambda x, s: 1
        if f is None:
            f = lambda y: y
        self.c = c
        self.d = d
        self.k = k
        self.F = f

        if lower_bound is None:
            lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.global_error_tolerance = global_error_tolerance

        self.interpolation_kind = interpolation_kind

        self.smoothing_factor = smoothing_factor

        self.max_iterations = max_iterations

        self.iteration = 0
        self.y = None
        self.wall_time_elapsed = None

    def global_error(self, y1, y2):
        diff = y1 - y2
        return np.sqrt(np.sum(diff * diff))

    def interpolate_y(self, y):
        return inter.interp1d(self.x, y, kind = self.interpolation_kind, fill_value = 'extrapolate', assume_sorted = True)

    def solve_ode(self, rhs):
        solver = self.ode_solver(rhs)
        solver.set_integrator('lsoda', atol = self.global_error_tolerance, rtol = 0)
        solver.set_initial_value(self.y_initial, self.x[0])

        soln = np.empty_like(self.x, dtype = self.dtype)
        soln[0] = self.y_initial

        for idx, x in enumerate(self.x[1:]):
            solver.integrate(x)
            soln[idx + 1] = solver.y

        return soln

    def initial_y(self):
        return self.solve_ode(self.c)

    def solve_rhs_with_known_y(self, y):
        interp_y = self.interpolate_y(y)

        def integral(x):
            r, err = integ.quadrature(
                lambda s: self.k(x, s) * self.F(interp_y(s)),
                self.lower_bound(x),
                self.upper_bound(x),
                tol = 0,
                rtol = self.global_error_tolerance,
            )

            return r

        def rhs(x, y):
            return self.c(x, interp_y(x)) + (self.d(x) * integral(x))

        return self.solve_ode(rhs)

    def next_curr(self, curr, guess):
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    def solve(self):
        self.iteration = 0
        y_curr = self.initial_y()
        y_guess = self.solve_rhs_with_known_y(y_curr)

        curr_error = self.global_error(y_curr, y_guess)

        while curr_error > self.global_error_tolerance and (self.max_iterations is None or self.iteration < self.max_iterations):
            new_curr = self.next_curr(y_curr, y_guess)
            new_guess = self.solve_rhs_with_known_y(new_curr)

            new_error = self.global_error(new_curr, new_guess)
            if new_error > curr_error:
                warnings.warn(f'Error increased on iteration {self.iteration}', ConvergenceWarning)

            y_curr = new_curr
            y_guess = new_guess
            curr_error = new_error

            self.iteration += 1
            logger.debug(f'Advanced to iteration {self.iteration}. Current error: {curr_error}.')

        self.y = self.next_curr(y_curr, y_guess)
        return self.y


class CIDESolver(IDESolver):
    dtype = np.complex128
    ode_solver = integ.complex_ode

    def global_error(self, y1, y2):
        diff = y1 - y2
        return np.sqrt(np.real(np.sum(np.abs(diff * diff))))
