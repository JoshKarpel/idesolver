"""
A general purpose integro-differential equation (IDE) solver.

Copyright (C) 2017  Joshua T Karpel
Full license available at github.com/JoshKarpel/LICENSE
"""

from typing import Union, Optional, Callable
import warnings
import logging

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter

logger = logging.getLogger('idesolver')
logger.setLevel(logging.DEBUG)


class ConvergenceWarning(Warning):
    pass


class IDESolverException(Exception):
    pass


class InvalidParameter(IDESolverException):
    pass


def complex_quadrature(integrand, a, b, **kwargs):
    """A thin wrapper over `scipy.integrate.quadrature` that handles splitting the real and complex parts of the integral and recombining them."""

    def real_func(x):
        return np.real(integrand(x))

    def imag_func(x):
        return np.imag(integrand(x))

    real_integral = integ.quadrature(real_func, a, b, **kwargs)
    imag_integral = integ.quadrature(imag_func, a, b, **kwargs)

    return real_integral[0] + (1j * imag_integral[0]), real_integral[1], imag_integral[1]


class IDESolver:
    """
    A class that handles solving an integro-differential equation of the form

    .. math::

        \\frac{dy}{dx} & = c(y, x) + d(x) \\int_{\\alpha(x)}^{\\beta(x)} k(x, s) \\, F( y(s) ) \\, ds, \\\\
        & x \\in [a, b], \\quad y(a) = y_0.

    """

    dtype = np.float64
    ode_solver = integ.ode

    def __init__(self,
                 x: np.ndarray,
                 y_0: float,
                 c: Optional[Callable] = None,
                 d: Optional[Callable] = None,
                 k: Optional[Callable] = None,
                 f: Optional[Callable] = None,
                 lower_bound: Optional[Callable] = None,
                 upper_bound: Optional[Callable] = None,
                 global_error_tolerance: float = 1e-9,
                 interpolation_kind: str = 'cubic',
                 max_iterations: Optional[int] = None,
                 smoothing_factor: float = .5,
                 store_intermediate: bool = False):
        """
        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The array of :math:`x` values to find the solution :math:`y(x)` at. Generally something like ``numpy.linspace(a, b, num_pts)``.
        y_0 : :class:`float`
            The initial condition, :math:`y_0 = y(a)`.
        c : callable
            The function :math:`c(y, x)`.
        d : callable
            The function :math:`d(x)`.
        k : callable
            The kernel function :math:`k(x, s)`.
        f : callable
            The function :math:`F(y)`.
        lower_bound : callable
            The lower bound function :math:`\\alpha(x)`.
        upper_bound : callable
            The upper bound function :math:`\\beta(x)`.
        global_error_tolerance : :class:`float`
        interpolation_kind : :class:`str`
            The type of interpolation to use. As the `kind` option of :class:`scipy.interpolate.interp1d`. Defaults to ``'cubic'``.
        max_iterations : :class:`int`
            The maximum number of iterations to use. If ``None``, iteration will not stop unless the `global_error_tolerance` is satisfied. Defaults to ``None``.
        smoothing_factor : :class:`float`
            The smoothing factor used to combine the current guess with the new guess at each iteration. Defaults to ``0.5``.
        """
        self.y_0 = y_0
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

        if not 0 < self.smoothing_factor < 1:
            raise InvalidParameter('Smoothing factor must be between 0 and 1')
        self.smoothing_factor = smoothing_factor

        if max_iterations is not None and max_iterations <= 0:
            raise InvalidParameter('If given, max iterations must be greater than 0')
        self.max_iterations = max_iterations

        self.iteration = 0
        self.y = None

        self.store_intermediate = store_intermediate
        if self.store_intermediate:
            self.y_intermediate = {}

    def solve(self) -> np.ndarray:
        """
        Compute the solution to the IDE.

        The solution is returned, and also stored in the attribute ``y``.

        Will emit warning messages if the global error increases on an iteration.
        This does not necessarily mean that the algorithm is not converging, but may indicate that it's having problems.

        Returns
        -------
        The solution to the IDE.
        """
        self.iteration = 0

        y_curr = self.initial_y()
        y_guess = self.solve_rhs_with_known_y(y_curr)
        current_error = self.global_error(y_curr, y_guess)

        while current_error > self.global_error_tolerance and (self.max_iterations is None or self.iteration < self.max_iterations):
            if self.store_intermediate:
                self.y_intermediate[self.iteration] = y_curr

            new_current = self.next_y(y_curr, y_guess)
            new_guess = self.solve_rhs_with_known_y(new_current)
            new_error = self.global_error(new_current, new_guess)
            if new_error > current_error:
                warnings.warn(f'Error increased on iteration {self.iteration}', ConvergenceWarning)

            y_curr, y_guess, current_error = new_current, new_guess, new_error

            self.iteration += 1

            logger.debug(f'Advanced to iteration {self.iteration}. Current error: {current_error}.')

        self.y = self.next_y(y_curr, y_guess)
        return self.y

    def initial_y(self) -> np.ndarray:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        return self.solve_ode(self.c)

    def next_y(self, curr: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    def global_error(self, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Return the global error estimate between `y1` and `y2`.

        The estimate is the square root of the sum of squared differences between `y1` and `y2`.

        Parameters
        ----------
        y1
            A guess of the solution.
        y2
            Another guess of the solution.

        Returns
        -------
        error :
            The global error estimate between `y1` and `y2`.
        """
        diff = y1 - y2
        return np.sqrt(np.sum(diff * diff))

    def solve_ode(self, rhs: Callable) -> np.ndarray:
        """Solves an ODE with the given right-hand side."""
        # TODO: update to use new scipy 1.0.0-style integrators
        solver = self.ode_solver(rhs)
        solver.set_integrator(
            'lsoda',
            atol = self.global_error_tolerance,
            rtol = 0,
        )
        solver.set_initial_value(self.y_0, self.x[0])

        soln = np.empty_like(self.x, dtype = self.dtype)
        soln[0] = self.y_0

        for idx, x in enumerate(self.x[1:], start = 1):
            solver.integrate(x)
            soln[idx] = solver.y

        return soln

    def solve_rhs_with_known_y(self, y: np.ndarray) -> np.ndarray:
        interp_y = self.interpolate_y(y)

        def integral(x):
            r, err = integ.quadrature(
                lambda s: self.k(x, s) * self.F(interp_y(s)),
                self.lower_bound(x),
                self.upper_bound(x),
                # maxiter = len(self.x),
                tol = 0,
                rtol = self.global_error_tolerance,
            )

            return r

        def rhs(x, y):
            return self.c(x, interp_y(x)) + (self.d(x) * integral(x))

        return self.solve_ode(rhs)

    def interpolate_y(self, y: np.ndarray) -> inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y
            The guess to interpolate.

        Returns
        -------
        interpolator :
            The interpolated function.
        """
        return inter.interp1d(
            x = self.x,
            y = y,
            kind = self.interpolation_kind,
            fill_value = 'extrapolate',
            assume_sorted = True,
        )


class CIDESolver(IDESolver):
    """
    This class uses a different set of solvers and a definition of the global error function appropriate for complex :math:`y`.
    Because it needs to use a complex-valued data type and a complex-valued ODE solver, this solver is slower than the real-valued :class:`IDESolver`.
    """

    dtype = np.complex128
    ode_solver = integ.complex_ode

    def global_error(self, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        diff = y1 - y2
        return np.sqrt(np.real(np.sum(np.abs(diff * diff))))
