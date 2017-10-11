import warnings
import datetime

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter


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
                 smoothing_factor = .5):
        self.y_initial = y_initial
        self.x = np.array(x)

        if c is None:
            c = lambda *_: 0
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

        self.smoothing_factor = smoothing_factor

        self.iteration = 0
        self.y = None
        self.wall_time_elapsed = None

    def initial_guess(self):
        solver = integ.complex_ode(self.c)
        solver.set_integrator('lsoda', atol = self.global_error_tolerance, rtol = 0)
        solver.set_initial_value(self.y_initial, self.x[0])

        soln = np.empty_like(self.x, dtype = np.complex128)
        soln[0] = self.y_initial

        for idx, x in enumerate(self.x[1:]):
            solver.integrate(x)
            soln[idx + 1] = solver.y

        return soln

    def global_error(self, y1, y2):
        diff = y1 - y2
        return np.sqrt(np.real(np.sum(np.abs(diff * diff))))

    def solve_rhs_with_known_y(self, y):
        interp_y = inter.interp1d(self.x, y, kind = 'cubic', fill_value = 'extrapolate', assume_sorted = True)

        def integral(x):
            r, re_err, im_err = complex_quadrature(
                lambda s: self.k(x, s) * self.F(interp_y(s)),
                self.lower_bound(x),
                self.upper_bound(x),
                tol = 0,
                rtol = self.global_error_tolerance,
            )

            return r

        def rhs(x, y):
            return self.c(x, interp_y(x)) + (self.d(x) * integral(x))

        solver = integ.complex_ode(rhs)
        solver.set_integrator('lsoda', atol = self.global_error_tolerance, rtol = 0)
        solver.set_initial_value(self.y_initial, self.x[0])

        soln = np.empty_like(self.x, dtype = np.complex128)
        soln[0] = self.y_initial

        for idx, x in enumerate(self.x[1:]):
            solver.integrate(x)
            soln[idx + 1] = solver.y

        return soln

    def next_curr(self, curr, guess):
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    def solve(self):
        self.wall_time_elapsed = None
        t = datetime.datetime.now()

        self.iteration = 1
        curr = self.initial_guess()
        guess = self.solve_rhs_with_known_y(curr)

        curr_error = self.global_error(curr, guess)

        while curr_error > self.global_error_tolerance:
            self.iteration += 1
            print('ITERATION', self.iteration, self.global_error(curr, guess))
            new_curr = self.next_curr(curr, guess)
            new_guess = self.solve_rhs_with_known_y(new_curr)

            new_error = self.global_error(new_curr, new_guess)
            if new_error > curr_error:
                warnings.warn(f'Error increased on iteration {self.iteration}', ConvergenceWarning)
                # break

            curr = new_curr
            guess = new_guess
            curr_error = new_error

        self.wall_time_elapsed = datetime.datetime.now() - t
        self.y = guess
        return guess
