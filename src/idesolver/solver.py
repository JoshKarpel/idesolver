import functools
import time

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter


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
                 global_error = 1e-8,
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

        self.global_error = global_error

        self.smoothing_factor = smoothing_factor

        self.iterations = 0

    def initial_guess(self):
        return integ.odeint(
            self.c,
            self.y_initial,
            self.x,
        )[:, 0]

    def compare(self, y1, y2):
        diff = y1 - y2
        return np.sum(diff * diff)

    def solve_rhs_with_known_y(self, y):
        interp_y = inter.interp1d(self.x, y, fill_value = 'extrapolate')

        def s(x):
            return np.linspace(self.lower_bound(x), self.upper_bound(x), 100)

        return integ.odeint(
            lambda _, x: self.c(interp_y(x), x) + (self.d(x) * integ.simps(
                y = self.k(x, s(x)) * self.F(interp_y(s(x))),
                x = s(x),
            )),
            self.y_initial,
            self.x,
        )[:, 0]

    def next_curr(self, curr, guess):
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    def solve(self):
        t = time.time()

        self.iterations = 1
        curr = self.initial_guess()
        guess = self.solve_rhs_with_known_y(curr)

        while self.compare(curr, guess) > self.global_error:
            self.iterations += 1
            curr = self.next_curr(curr, guess)
            guess = self.solve_rhs_with_known_y(curr)

        self.y = guess
        print(time.time() - t)
        return guess
