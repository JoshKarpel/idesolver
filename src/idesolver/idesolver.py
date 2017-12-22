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


class IDEConvergenceWarning(Warning):
    pass


class IDESolverException(Exception):
    pass


class InvalidParameter(IDESolverException):
    pass


class ODESolutionFailed(IDESolverException):
    pass


class UnexpectedlyComplexValuedIDE(IDESolverException):
    pass


def complex_quad(integrand, a, b, **kwargs):
    """A thin wrapper over `scipy.integrate.quadrature` that handles splitting the real and complex parts of the integral and recombining them."""

    def real_func(x):
        return np.real(integrand(x))

    def imag_func(x):
        return np.imag(integrand(x))

    real_integral = integ.quad(real_func, a, b, **kwargs)
    imag_integral = integ.quad(imag_func, a, b, **kwargs)

    return real_integral[0] + (1j * imag_integral[0]), real_integral[1], imag_integral[1]


_COMPLEX_NUMERIC_TYPES = [complex, np.complex128]


class IDESolver:
    """
    A class that handles solving an integro-differential equation of the form

    .. math::

        \\frac{dy}{dx} & = c(y, x) + d(x) \\int_{\\alpha(x)}^{\\beta(x)} k(x, s) \\, F( y(s) ) \\, ds, \\\\
        & x \\in [a, b], \\quad y(a) = y_0.

    """

    def __init__(self,
                 x: np.ndarray,
                 y_0: Union[float, np.float64, complex, np.complex128],
                 c: Optional[Callable] = None,
                 d: Optional[Callable] = None,
                 k: Optional[Callable] = None,
                 f: Optional[Callable] = None,
                 lower_bound: Optional[Callable] = None,
                 upper_bound: Optional[Callable] = None,
                 global_error_tolerance: float = 1e-6,
                 max_iterations: Optional[int] = None,
                 ode_method = 'RK45',
                 ode_atol = 1e-8,
                 ode_rtol = 1e-8,
                 int_atol = 1e-8,
                 int_rtol = 1e-8,
                 interpolation_kind: str = 'cubic',
                 smoothing_factor: float = .5,
                 store_intermediate_y: bool = False):
        """
        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The array of :math:`x` values to find the solution :math:`y(x)` at. Generally something like ``numpy.linspace(a, b, num_pts)``.
        y_0 : :class:`float` or :class:`complex`
            The initial condition, :math:`y_0 = y(a)`.
        c :
            The function :math:`c(y, x)`.
        d :
            The function :math:`d(x)`.
        k :
            The kernel function :math:`k(x, s)`.
        f :
            The function :math:`F(y)`.
        lower_bound : callable
            The lower bound function :math:`\\alpha(x)`.
        upper_bound : callable
            The upper bound function :math:`\\beta(x)`.
        global_error_tolerance : :class:`float`
            The algorithm will continue until the global errors goes below this or uses more than `max_iterations` iterations.
        interpolation_kind : :class:`str`
            The type of interpolation to use. As the `kind` option of :class:`scipy.interpolate.interp1d`. Defaults to ``'cubic'``.
        max_iterations : :class:`int`
            The maximum number of iterations to use. If ``None``, iteration will not stop unless the `global_error_tolerance` is satisfied. Defaults to ``None``.
        smoothing_factor : :class:`float`
            The smoothing factor used to combine the current guess with the new guess at each iteration. Defaults to ``0.5``.
        """
        if type(y_0) in _COMPLEX_NUMERIC_TYPES:
            self.integrator = complex_quad
        else:
            self.integrator = integ.quad
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

        if not 0 < smoothing_factor < 1:
            raise InvalidParameter('Smoothing factor must be between 0 and 1')
        self.smoothing_factor = smoothing_factor

        if max_iterations is not None and max_iterations <= 0:
            raise InvalidParameter('If given, max iterations must be greater than 0')
        self.max_iterations = max_iterations

        self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        self.int_atol = int_atol
        self.int_rtol = int_rtol

        self.store_intermediate = store_intermediate_y
        if self.store_intermediate:
            self.y_intermediate = []

        self.iteration = None
        self.y = None
        self.global_error = None

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
        # check if the user messed up by not passing y_0 as a complex number when they should have
        with warnings.catch_warnings():
            warnings.filterwarnings(action = 'error', message = 'Casting complex values', category = np.ComplexWarning)

            try:
                self.iteration = 0

                y_curr = self._initial_y()
                y_guess = self._solve_rhs_with_known_y(y_curr)
                current_error = self._global_error(y_curr, y_guess)

                while current_error > self.global_error_tolerance:
                    if self.store_intermediate:
                        self.y_intermediate.append(y_curr)

                    new_current = self._next_y(y_curr, y_guess)
                    new_guess = self._solve_rhs_with_known_y(new_current)
                    new_error = self._global_error(new_current, new_guess)
                    if new_error > current_error:
                        warnings.warn(f'Error increased on iteration {self.iteration}', IDEConvergenceWarning)

                    y_curr, y_guess, current_error = new_current, new_guess, new_error

                    self.iteration += 1

                    logger.debug(f'Advanced to iteration {self.iteration}. Current error: {current_error}.')

                    if self.max_iterations is not None and self.iteration >= self.max_iterations:
                        warnings.warn(IDEConvergenceWarning(f'Used maximum number of iterations ({self.max_iterations}), but only got to global error {current_error} (target {self.global_error_tolerance})'))
                        break
            except np.ComplexWarning:
                raise UnexpectedlyComplexValuedIDE('Detected complex-valued IDE. Make sure to pass y_0 as a complex number.')

        # self.y = self._next_y(y_curr, y_guess)
        self.y = y_curr
        self.global_error = current_error

        return self.y

    def _initial_y(self) -> np.ndarray:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        return self._solve_ode(self.c)

    def _next_y(self, curr: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    def _global_error(self, y1: np.ndarray, y2: np.ndarray) -> float:
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
        return np.sqrt(np.real(np.vdot(diff, diff)))

    def _solve_rhs_with_known_y(self, y: np.ndarray) -> np.ndarray:
        """Solves the right-hand-side of the IDE as if :math:`y` was `y`."""
        interp_y = self._interpolate_y(y)

        def integral(x):
            result, *_ = self.integrator(
                lambda s: self.k(x, s) * self.F(interp_y(s)),
                self.lower_bound(x),
                self.upper_bound(x),
                epsabs = self.int_atol,
                epsrel = self.int_rtol,
            )
            return result

        def rhs(x, y):
            return self.c(x, interp_y(x)) + (self.d(x) * integral(x))

        return self._solve_ode(rhs)

    def _interpolate_y(self, y: np.ndarray) -> inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y
            The y values to interpolate (probably a guess at the solution).

        Returns
        -------
        interpolator :
            The interpolator function.
        """
        return inter.interp1d(
            x = self.x,
            y = y,
            kind = self.interpolation_kind,
            fill_value = 'extrapolate',
            assume_sorted = True,
        )

    def _solve_ode(self, rhs: Callable) -> np.ndarray:
        """Solves an ODE with the given right-hand side."""
        sol = integ.solve_ivp(
            fun = rhs,
            y0 = np.array([self.y_0]),
            t_span = (self.x[0], self.x[-1]),
            t_eval = self.x,
            method = self.ode_method,
            atol = self.ode_atol,
            rtol = self.ode_rtol,
        )

        if not sol.success:
            raise ODESolutionFailed(f'Error while trying to solve ODE: {sol.status}')

        return sol.y[0]
