import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter


class IDESolver:
    def __init__(self,
                 y0,
                 x,
                 c,
                 d,
                 k,
                 alpha,
                 beta,
                 F,
                 tol = 1e-8):
        self.x = x
        self.c = c
        self.d = d
        self.k = k
        self.F = F

        self.y0 = y0

        self.tol = tol

        self.iterations = 0

    def initial_guess(self):
        return integ.odeint(
            self.c,
            self.y0,
            self.x,
        )[:, 0]

    def compare(self, y1, y2):
        diff = y1 - y2
        return np.sum(diff * diff)

    def solve_rhs_with_known_y(self, y):
        s = self.x
        interp_y = inter.interp1d(self.x, y, fill_value = 'extrapolate')
        return integ.odeint(
            lambda _, x: self.c(interp_y(x), x) + (self.d(x) * integ.simps(
                y = self.k(x, s) * self.F(interp_y(s)),
                x = s,
            )),
            self.y0,
            self.x,
        )[:, 0]

    def next_curr(self, curr, guess):
        a = 0.5
        return (a * curr) + ((1 - a) * guess)

    def solve(self):
        self.iterations = 1
        curr = self.initial_guess()
        guess = self.solve_rhs_with_known_y(curr)

        while self.compare(curr, guess) > self.tol:
            self.iterations += 1
            curr = self.next_curr(curr, guess)
            guess = self.solve_rhs_with_known_y(curr)

        self.y = guess
        return guess
