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


def complex_quad(integrand: Callable, lower_bound: float, upper_bound: float, **kwargs) -> (complex, float, float, tuple, tuple):
    """
    A thin wrapper over :func:`scipy.integrate.quad` that handles splitting the real and complex parts of the integral and recombining them.
    Keyword arguments are passed to both of the internal ``quad`` calls.
    """
    real_result, real_error, *real_extra = integ.quad(lambda x: np.real(integrand(x)),
                                                      lower_bound, upper_bound, **kwargs)
    imag_result, imag_error, *imag_extra = integ.quad(lambda x: np.imag(integrand(x)),
                                                      lower_bound, upper_bound, **kwargs)

    return real_result + (1j * imag_result), real_error, imag_error, real_extra, imag_extra


# types to recognize as complex in y_0
_COMPLEX_NUMERIC_TYPES = [complex, np.complex128]


class IDESolver:
    """
    A class that handles solving an integro-differential equation of the form

    .. math::

        \\frac{dy}{dx} & = c(y, x) + d(x) \\int_{\\alpha(x)}^{\\beta(x)} k(x, s) \\, F( y(s) ) \\, ds, \\\\
        & x \\in [a, b], \\quad y(a) = y_0.

    Attributes
    ----------
    x : :class:`numpy.ndarray`
        The positions where the solution is calculated (i.e., where `y` is evaluated).
    y : :class:`numpy.ndarray`
        The solution :math:`y(x)`. ``None`` until :meth:`IDESolver.solve` is finished.
    global_error : :class:`float`
        The final global error estimate. ``None`` until :meth:`IDESolver.solve` is finished.
    iteration : :class:`int`
        The current iteration. ``None`` until :meth:`IDESolver.solve` starts.
    y_intermediate :
        The intermediate solutions. Only exists if `store_intermediate_y` is ``True``.

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
                 ode_method: str = 'RK45',
                 ode_atol: float = 1e-8,
                 ode_rtol: float = 1e-8,
                 int_atol: float = 1e-8,
                 int_rtol: float = 1e-8,
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
        lower_bound :
            The lower bound function :math:`\\alpha(x)`.
        upper_bound :
            The upper bound function :math:`\\beta(x)`.
        global_error_tolerance : :class:`float`
            The algorithm will continue until the global errors goes below this or uses more than `max_iterations` iterations. If ``None``, the algorithm continues until hitting `max_iterations`.
        max_iterations : :class:`int`
            The maximum number of iterations to use. If ``None``, iteration will not stop unless the `global_error_tolerance` is satisfied. Defaults to ``None``.
        ode_method : :class:`str`
            The ODE solution method to use. As the `method` option of :func:`scipy.integrate.solve_ivp`. Defaults to ``'RK45'``, which is good for non-stiff systems.
        ode_atol : :class:`float`
            The absolute tolerance for the ODE solver. As the `atol` argument of :func:`scipy.integrate.solve_ivp`.
        ode_rtol : :class:`float`
            The relative tolerance for the ODE solver. As the `rtol` argument of :func:`scipy.integrate.solve_ivp`.
        int_atol : :class:`float`
            The absolute tolerance for the integration routine. As the `epsabs` argument of :func:`scipy.integrate.quad`.
        int_rtol : :class:`float`
            The relative tolerance for the integration routine. As the `epsrel` argument of :func:`scipy.integrate.quad`.
        interpolation_kind : :class:`str`
            The type of interpolation to use. As the `kind` argument of :class:`scipy.interpolate.interp1d`. Defaults to ``'cubic'``.
        smoothing_factor : :class:`float`
            The smoothing factor used to combine the current guess with the new guess at each iteration. Defaults to ``0.5``.
        store_intermediate_y : :class:`bool`
            If ``True``, the intermediate guesses for :math:`y(x)` at each iteration will be stored in the attribute `y_intermediate`.
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

        if global_error_tolerance == 0 and max_iterations is None:
            raise InvalidParameter('global_error_tolerance cannot be zero if max_iterations is None')
        if global_error_tolerance < 0:
            raise InvalidParameter('global_error_tolerance cannot be negative')
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

    def solve(self, callback: Optional[Callable] = None) -> np.ndarray:
        """
        Compute the solution to the IDE.

        The solution is returned, and also stored in the attribute ``y``.

        Will emit warning messages if the global error increases on an iteration.
        This does not necessarily mean that the algorithm is not converging, but may indicate that it's having problems.

        Will emit a warning message if the maximum number of iterations is used without reaching the global error tolerance.

        Parameters
        ----------
        callback :
            A function to call after each iteration. The function is passed the :class:`IDESolver` instance, the current :math:`y` guess, and the current global error.

        Returns
        -------
        :class:`numpy.ndarray`
            The solution to the IDE (i.e., :math:`y(x)`).
        """
        # check if the user messed up by not passing y_0 as a complex number when they should have
        with warnings.catch_warnings():
            warnings.filterwarnings(action = 'error', message = 'Casting complex values', category = np.ComplexWarning)

            try:
                y_current = self._initial_y()
                y_guess = self._solve_rhs_with_known_y(y_current)
                error_current = self._global_error(y_current, y_guess)
                if self.store_intermediate:
                    self.y_intermediate.append(y_current)

                self.iteration = 0

                logger.debug(f'Advanced to iteration {self.iteration}. Current error: {error_current}.')
                if callback is not None:
                    logger.debug(f'Calling {callback} after iteration {self.iteration}')
                    callback(self, y_guess, error_current)

                while error_current > self.global_error_tolerance:

                    new_current = self._next_y(y_current, y_guess)
                    new_guess = self._solve_rhs_with_known_y(new_current)
                    new_error = self._global_error(new_current, new_guess)
                    if new_error > error_current:
                        warnings.warn(f'Error increased on iteration {self.iteration}', IDEConvergenceWarning)

                    y_current, y_guess, error_current = new_current, new_guess, new_error

                    if self.store_intermediate:
                        self.y_intermediate.append(y_current)

                    self.iteration += 1

                    logger.debug(f'Advanced to iteration {self.iteration}. Current error: {error_current}.')

                    if callback is not None:
                        logger.debug(f'Calling {callback} after iteration {self.iteration}')
                        callback(self, y_guess, error_current)

                    if self.max_iterations is not None and self.iteration >= self.max_iterations:
                        warnings.warn(IDEConvergenceWarning(f'Used maximum number of iterations ({self.max_iterations}), but only got to global error {error_current} (target {self.global_error_tolerance})'))
                        break
            except np.ComplexWarning:
                raise UnexpectedlyComplexValuedIDE('Detected complex-valued IDE. Make sure to pass y_0 as a complex number.')

        self.y = y_guess
        self.global_error = error_current

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
        y1 : :class:`numpy.ndarray`
            A guess of the solution.
        y2 : :class:`numpy.ndarray`
            Another guess of the solution.

        Returns
        -------
        error : :class:`float`
            The global error estimate between `y1` and `y2`.
        """
        diff = y1 - y2
        return np.sqrt(np.real(np.vdot(diff, diff)))

    def _solve_rhs_with_known_y(self, y: np.ndarray) -> np.ndarray:
        """Solves the right-hand-side of the IDE as if :math:`y` was `y`."""
        interpolated_y = self._interpolate_y(y)

        def integral(x):
            result, *_ = self.integrator(
                lambda s: self.k(x, s) * self.F(interpolated_y(s)),
                self.lower_bound(x),
                self.upper_bound(x),
                epsabs = self.int_atol,
                epsrel = self.int_rtol,
            )
            return result

        def rhs(x, y):
            return self.c(x, interpolated_y(x)) + (self.d(x) * integral(x))

        return self._solve_ode(rhs)

    def _interpolate_y(self, y: np.ndarray) -> inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            The y values to interpolate (probably a guess at the solution).

        Returns
        -------
        interpolator : :class:`scipy.interpolate.interp1d`
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