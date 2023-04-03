from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union

from numpy import (
    ComplexWarning,
    array,
    complex128,
    complex_,
    float64,
    float_,
    imag,
    ndarray,
    real,
    sqrt,
    vdot,
    zeros_like,
)
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

from idesolver import exceptions

logger = logging.getLogger("idesolver")
logger.setLevel(logging.DEBUG)


def complex_quad(
    integrand: Callable, lower_bound: float, upper_bound: float, **kwargs
) -> (complex, float, float, tuple, tuple):
    """
    A thin wrapper over [`scipy.integrate.quad`][scipy.integrate.quad]
    that handles splitting the real and complex parts of the integral and recombining them.
    Keyword arguments are passed to both of the internal [`scipy.integrate.quad`][scipy.integrate.quad] calls.
    """
    real_result, real_error, *real_extra = quad(
        lambda x: real(integrand(x)), lower_bound, upper_bound, **kwargs
    )
    imag_result, imag_error, *imag_extra = quad(
        lambda x: imag(integrand(x)), lower_bound, upper_bound, **kwargs
    )

    return (
        real_result + (1j * imag_result),
        real_error,
        imag_error,
        real_extra,
        imag_extra,
    )


FloatsOrComplexes = Union[NDArray[float_], NDArray[complex_]]


def global_error(y1: FloatsOrComplexes, y2: FloatsOrComplexes) -> float:
    """
    The default global error function.

    The estimate is the square root of the sum of squared differences between `y1` and `y2`.

    Parameters:
        y1: A guess of the solution.
        y2: Another guess of the solution.

    Returns:
        The global error estimate between `y1` and `y2`.
    """
    diff = y1 - y2
    return sqrt(real(vdot(diff, diff)))


def coerce_to_array(
    to_coerce: Union[float, float64, complex, complex128, ndarray, list]
) -> ndarray:
    """Coerce `to_coerce` into a numpy array"""
    return array(to_coerce, ndmin=1, copy=False)


def dtype(n):
    return n.dtype if isinstance(n, ndarray) else type(n)


# data types to recognize as complex in y_0
_COMPLEX_NUMERIC_TYPES = [complex, complex128]

Y = TypeVar("Y", bound=Union[float_, complex_])


@dataclass(frozen=True)
class SolveIDEResult(Generic[Y]):
    ide: IDE
    y: NDArray[Y]
    global_error: float_
    iterations: int


class IDE(Generic[Y]):
    """
    Attributes:
        x:
            The array of $x$ values to find the solution $y(x)$ at.
            Generally something like `numpy.linspace(a, b, num_pts)`.
        y_0:
            The initial condition, $y_0 = y(a)$ (can be multidimensional).
        c:
            The function $c(y, x)$.
            Defaults to $c(y, x) = 0$.
        d:
            The function $d(x)$.
            Defaults to $d(x) = 1$.
        k:
            The kernel function $k(x, s)$.
            Defaults to $k(x, s) = 1$.
        f:
            The function $F(y)$.
            Defaults to $f(y) = 0$.
        lower_bound:
            The lower bound function $\\alpha(x)$.
            Defaults to the first element of `x`.
        upper_bound:
            The upper bound function $\\beta(x)$.
            Defaults to the last element of `x`.
    """

    def __init__(
        self,
        x: ndarray,
        y_0: Union[float, float64, complex, complex128, ndarray, list],
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        k: Optional[Callable] = None,
        f: Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
    ):
        self.y_0 = coerce_to_array(y_0)
        zeros_like_y_0 = zeros_like(self.y_0)

        if dtype(self.y_0) in _COMPLEX_NUMERIC_TYPES:
            self.integrator = complex_quad
        else:
            self.integrator = quad

        self.x = array(x)

        if c is None:
            c = lambda x, y: zeros_like_y_0.copy()
        if d is None:
            d = lambda x: 1
        if k is None:
            k = lambda x, s: 1
        if f is None:
            f = lambda y: zeros_like_y_0.copy()

        self.c = lambda x, y: coerce_to_array(c(x, y))
        self.d = lambda x: coerce_to_array(d(x))
        self.k = lambda x, s: coerce_to_array(k(x, s))
        self.f = lambda y: coerce_to_array(f(y))

        if lower_bound is None:
            lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


def solve_ide(
    ide: IDE,
    global_error_tolerance: float = 1e-6,
    max_iterations: Optional[int] = None,
    ode_method: str = "RK45",
    ode_atol: float = 1e-8,
    ode_rtol: float = 1e-8,
    int_atol: float = 1e-8,
    int_rtol: float = 1e-8,
    interpolation_kind: str = "cubic",
    smoothing_factor: float = 0.5,
    global_error_function: Callable = global_error,
    callback=None,
) -> SolveIDEResult:
    r"""
    A class that handles solving an integro-differential equation of the form

    $$
    \begin{aligned}
        \frac{dy}{dx} & = c(y, x) + d(x) \int_{\alpha(x)}^{\beta(x)} k(x, s) \, f( y(s) ) \, ds, \\
        & x \in [a, b], \quad y(a) = y_0.
    \end{aligned}
    $$

    Arguments:
        ide:
            The IDE to solve.
        global_error_tolerance:
            The algorithm will continue until the global errors goes below this or uses more than `max_iterations` iterations. If `None`, the algorithm continues until hitting `max_iterations`.
        max_iterations:
            The maximum number of iterations to use. If `None`, iteration will not stop unless the `global_error_tolerance` is satisfied. Defaults to `None`.
        ode_method:
            The ODE solution method to use. As the `method` option of :func:`scipy.integrate.solve_ivp`. Defaults to `'RK45'`, which is good for non-stiff systems.
        ode_atol:
            The absolute tolerance for the ODE solver.
            As the `atol` argument of :func:`scipy.integrate.solve_ivp`.
        ode_rtol:
            The relative tolerance for the ODE solver.
            As the `rtol` argument of :func:`scipy.integrate.solve_ivp`.
        int_atol:
            The absolute tolerance for the integration routine. As the `epsabs` argument of :func:`scipy.integrate.quad`.
        int_rtol:
            The relative tolerance for the integration routine. As the `epsrel` argument of :func:`scipy.integrate.quad`.
        interpolation_kind:
            The type of interpolation to use. As the `kind` argument of :class:`scipy.interpolate.interp1d`. Defaults to `'cubic'`.
        smoothing_factor:
            The smoothing factor used to combine the current guess with the new guess at each iteration. Defaults to `0.5`.
        global_error_function:
            The function to use to calculate the global error. Defaults to :func:`global_error`.
    """
    y_0 = coerce_to_array(ide.y_0)

    if dtype(y_0) in _COMPLEX_NUMERIC_TYPES:
        integrator = complex_quad
    else:
        integrator = quad

    if global_error_tolerance == 0 and max_iterations is None:
        raise exceptions.InvalidParameter(
            "global_error_tolerance cannot be 0 if max_iterations is None"
        )
    if global_error_tolerance < 0:
        raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")

    if not 0 < smoothing_factor < 1:
        raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")

    if max_iterations is not None and max_iterations <= 0:
        raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")

    # check if the user messed up by not passing y_0 as a complex number when they should have
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="error",
            message="Casting complex values",
            category=ComplexWarning,
        )

        try:
            # Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE.
            y_current = _solve_ode(
                rhs=ide.c,
                y_0=y_0,
                x=ide.x,
                ode_method=ode_method,
                ode_atol=ode_atol,
                ode_rtol=ode_rtol,
            )

            y_guess = _solve_rhs_with_known_y(
                ide=ide,
                y=y_current,
                integrator=integrator,
                int_atol=int_atol,
                int_rtol=int_rtol,
                ode_method=ode_method,
                ode_atol=ode_atol,
                ode_rtol=ode_rtol,
                interpolation_kind=interpolation_kind,
            )
            error_current = global_error_function(y_current, y_guess)

            iteration = 0

            logger.debug(f"Advanced to iteration {iteration}. Current error: {error_current}.")
            if callback is not None:
                logger.debug(f"Calling {callback} after iteration {iteration}")
                callback(y_guess, error_current)

            while error_current > global_error_tolerance:
                new_current = _next_y(
                    y_current,
                    y_guess,
                    smoothing_factor=smoothing_factor,
                )
                new_guess = _solve_rhs_with_known_y(
                    ide=ide,
                    y=new_current,
                    integrator=integrator,
                    int_atol=int_atol,
                    int_rtol=int_rtol,
                    ode_method=ode_method,
                    ode_atol=ode_atol,
                    ode_rtol=ode_rtol,
                    interpolation_kind=interpolation_kind,
                )
                new_error = global_error_function(new_current, new_guess)
                if new_error > error_current:
                    warnings.warn(
                        f"Error increased on iteration {iteration}",
                        exceptions.IDEConvergenceWarning,
                    )

                y_current, y_guess, error_current = (
                    new_current,
                    new_guess,
                    new_error,
                )

                iteration += 1

                logger.debug(f"Advanced to iteration {iteration}. Current error: {error_current}.")

                if callback is not None:
                    logger.debug(f"Calling {callback} after iteration {iteration}")
                    callback(y_guess, error_current)

                if max_iterations is not None and iteration >= max_iterations:
                    warnings.warn(
                        exceptions.IDEConvergenceWarning(
                            f"Used maximum number of iterations ({max_iterations}), but only got to global error {error_current} (target {global_error_tolerance})"
                        )
                    )
                    break
        except (ComplexWarning, TypeError) as e:
            raise exceptions.UnexpectedlyComplexValuedIDE(
                "Detected complex-valued IDE. Make sure to pass y_0 as a complex number."
            ) from e

    y = y_guess
    global_error = error_current

    # get rid of the array wrapper if the dimension is 1
    if y_0.size == 1:
        y = y[0]

    return SolveIDEResult(
        ide=ide,
        y=y,
        global_error=global_error,
        iterations=iteration,
    )


def _next_y(curr: ndarray, guess: ndarray, smoothing_factor: float) -> ndarray:
    """Calculate the next guess at the solution by merging two guesses."""
    return (smoothing_factor * curr) + ((1 - smoothing_factor) * guess)


def _solve_rhs_with_known_y(
    ide: IDE,
    y: ndarray,
    integrator,
    int_atol: float,
    int_rtol: float,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
    interpolation_kind: str,
) -> ndarray:
    """Solves the right-hand-side of the IDE as if $y(x)$ was the fixed array `y`."""
    interpolated_y = _interpolate_y(x=ide.x, y=y, interpolation_kind=interpolation_kind)

    def integral(x):
        def integrand(s):
            return ide.k(x, s) * ide.f(interpolated_y(s))

        result = []
        for i in range(ide.y_0.size):
            r, *_ = integrator(
                lambda s: integrand(s)[i],
                ide.lower_bound(x),
                ide.upper_bound(x),
                epsabs=int_atol,
                epsrel=int_rtol,
            )
            result.append(r)
        return coerce_to_array(result)

    def rhs(x, y):
        return ide.c(x, interpolated_y(x)) + (ide.d(x) * integral(x))

    return _solve_ode(
        rhs=rhs, y_0=ide.y_0, x=ide.x, ode_method=ode_method, ode_atol=ode_atol, ode_rtol=ode_rtol
    )


def _interpolate_y(x: NDArray[float_], y: ndarray, interpolation_kind: str) -> interp1d:
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
    return interp1d(
        x=x,
        y=y,
        kind=interpolation_kind,
        fill_value="extrapolate",
        assume_sorted=True,
    )


def _solve_ode(
    rhs: Callable,
    y_0,
    x,
    ode_method: str,
    ode_atol: float,
    ode_rtol: float,
) -> ndarray:
    """Solves an ODE with the given right-hand side."""
    sol = solve_ivp(
        fun=rhs,
        y0=y_0,
        t_span=(x[0], x[-1]),
        t_eval=x,
        method=ode_method,
        atol=ode_atol,
        rtol=ode_rtol,
    )

    if not sol.success:
        raise exceptions.ODESolutionFailed(f"Error while trying to solve ODE: {sol.status}")

    return sol.y


# class IDESolver:
#     r"""
#
#
#     Attributes:
#         x: The positions where the solution is calculated (i.e., where $y$ is evaluated).
#         y: The solution $y(x)$. `None` until [`IDESolver.solve`][idesolver.IDESolver.solve] is finished.
#         global_error: The final global error estimate. `None` until [`IDESolver.solve`][idesolver.IDESolver.solve] is finished.
#         iteration: The current iteration. `None` until [`IDESolver.solve`][idesolver.IDESolver.solve] starts.
#         y_intermediate: The intermediate solutions. Only exists if `store_intermediate_y` is `True`.
#
#     """
#
#     def __init__(
#         self,
#         x: ndarray,
#         y_0: Union[float, float64, complex, complex128, ndarray, list],
#         c: Optional[Callable] = None,
#         d: Optional[Callable] = None,
#         k: Optional[Callable] = None,
#         f: Optional[Callable] = None,
#         lower_bound: Optional[Callable] = None,
#         upper_bound: Optional[Callable] = None,
#         global_error_tolerance: float = 1e-6,
#         max_iterations: Optional[int] = None,
#         ode_method: str = "RK45",
#         ode_atol: float = 1e-8,
#         ode_rtol: float = 1e-8,
#         int_atol: float = 1e-8,
#         int_rtol: float = 1e-8,
#         interpolation_kind: str = "cubic",
#         smoothing_factor: float = 0.5,
#         store_intermediate_y: bool = False,
#         global_error_function: Callable = global_error,
#     ):
#         self.y_0 = coerce_to_array(y_0)
#
#         if dtype(self.y_0) in _COMPLEX_NUMERIC_TYPES:
#             self.integrator = complex_quad
#         else:
#             self.integrator = integ.quad
#
#         self.x = array(x)
#
#         if c is None:
#             c = lambda x, y: self._zeros()
#         if d is None:
#             d = lambda x: 1
#         if k is None:
#             k = lambda x, s: 1
#         if f is None:
#             f = lambda y: self._zeros()
#
#         self.c = lambda x, y: coerce_to_array(c(x, y))
#         self.d = lambda x: coerce_to_array(d(x))
#         self.k = lambda x, s: coerce_to_array(k(x, s))
#         self.f = lambda y: coerce_to_array(f(y))
#
#         if lower_bound is None:
#             lower_bound = lambda x: self.x[0]
#         if upper_bound is None:
#             upper_bound = lambda x: self.x[-1]
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#
#         if global_error_tolerance == 0 and max_iterations is None:
#             raise exceptions.InvalidParameter(
#                 "global_error_tolerance cannot be 0 if max_iterations is None"
#             )
#         if global_error_tolerance < 0:
#             raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
#         self.global_error_tolerance = global_error_tolerance
#         self.global_error_function = global_error_function
#
#         self.interpolation_kind = interpolation_kind
#
#         if not 0 < smoothing_factor < 1:
#             raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")
#         self.smoothing_factor = smoothing_factor
#
#         if max_iterations is not None and max_iterations <= 0:
#             raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")
#         self.max_iterations = max_iterations
#
#         self.ode_method = ode_method
#         self.ode_atol = ode_atol
#         self.ode_rtol = ode_rtol
#
#         self.int_atol = int_atol
#         self.int_rtol = int_rtol
#
#         self.store_intermediate = store_intermediate_y
#         if self.store_intermediate:
#             self.y_intermediate = []
#
#         self.iteration = None
#         self.y = None
#         self.global_error = None
#
#     def _zeros(self) -> ndarray:
#         return zeros_like(self.y_0)
#
#     def solve(self, callback: Optional[Callable] = None) -> ndarray:
#         """
#         Compute the solution to the IDE.
#
#         Will emit a warning message if the global error increases on an iteration.
#         This does not necessarily mean that the algorithm is not converging, but may indicate that it's having problems.
#
#         Will emit a warning message if the maximum number of iterations is used without reaching the global error tolerance.
#
#         Parameters:
#             callback: A function to call after each iteration.
#                 The function is passed the [`IDESolver`][idesolver.IDESolver] instance, the current $y$ guess, and the current global error.
#
#         Returns:
#             The solution to the IDE (i.e., $y(x)$).
#         """
#         # check if the user messed up by not passing y_0 as a complex number when they should have
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 action="error",
#                 message="Casting complex values",
#                 category=ComplexWarning,
#             )
#
#             try:
#                 y_current = self._initial_y()
#                 y_guess = self._solve_rhs_with_known_y(y_current)
#                 error_current = self._global_error(y_current, y_guess)
#                 if self.store_intermediate:
#                     self.y_intermediate.append(y_current)
#
#                 self.iteration = 0
#
#                 logger.debug(
#                     f"Advanced to iteration {self.iteration}. Current error: {error_current}."
#                 )
#                 if callback is not None:
#                     logger.debug(f"Calling {callback} after iteration {self.iteration}")
#                     callback(self, y_guess, error_current)
#
#                 while error_current > self.global_error_tolerance:
#                     new_current = self._next_y(y_current, y_guess)
#                     new_guess = self._solve_rhs_with_known_y(new_current)
#                     new_error = self._global_error(new_current, new_guess)
#                     if new_error > error_current:
#                         warnings.warn(
#                             f"Error increased on iteration {self.iteration}",
#                             exceptions.IDEConvergenceWarning,
#                         )
#
#                     y_current, y_guess, error_current = (
#                         new_current,
#                         new_guess,
#                         new_error,
#                     )
#
#                     if self.store_intermediate:
#                         self.y_intermediate.append(y_current)
#
#                     self.iteration += 1
#
#                     logger.debug(
#                         f"Advanced to iteration {self.iteration}. Current error: {error_current}."
#                     )
#
#                     if callback is not None:
#                         logger.debug(f"Calling {callback} after iteration {self.iteration}")
#                         callback(self, y_guess, error_current)
#
#                     if self.max_iterations is not None and self.iteration >= self.max_iterations:
#                         warnings.warn(
#                             exceptions.IDEConvergenceWarning(
#                                 f"Used maximum number of iterations ({self.max_iterations}), but only got to global error {error_current} (target {self.global_error_tolerance})"
#                             )
#                         )
#                         break
#             except (ComplexWarning, TypeError) as e:
#                 raise exceptions.UnexpectedlyComplexValuedIDE(
#                     "Detected complex-valued IDE. Make sure to pass y_0 as a complex number."
#                 ) from e
#
#         self.y = y_guess
#         self.global_error = error_current
#
#         # get rid of the array wrapper if the dimension is 1
#         if self.y_0.size == 1:
#             self.y = self.y[0]
#             if self.store_intermediate:
#                 self.y_intermediate = [y[0] for y in self.y_intermediate]
#
#         return self.y
#
#     def _initial_y(self) -> ndarray:
#         """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
#         return self._solve_ode(self.c)
#
#     def _next_y(self, curr: ndarray, guess: ndarray) -> ndarray:
#         """Calculate the next guess at the solution by merging two guesses."""
#         return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)
#
#     def _global_error(self, y1: ndarray, y2: ndarray) -> float:
#         """
#         Return the global error estimate between `y1` and `y2`.
#
#         Parameters:
#             y1: A guess of the solution.
#             y2: Another guess of the solution.
#
#         Returns:
#             error: The global error estimate between `y1` and `y2`.
#         """
#         return self.global_error_function(y1, y2)
#
#     def _solve_rhs_with_known_y(self, y: ndarray) -> ndarray:
#         """Solves the right-hand-side of the IDE as if $y(x)$ was the fixed array `y`."""
#         interpolated_y = self._interpolate_y(y)
#
#         def integral(x):
#             def integrand(s):
#                 return self.k(x, s) * self.f(interpolated_y(s))
#
#             result = []
#             for i in range(self.y_0.size):
#                 r, *_ = self.integrator(
#                     lambda s: integrand(s)[i],
#                     self.lower_bound(x),
#                     self.upper_bound(x),
#                     epsabs=self.int_atol,
#                     epsrel=self.int_rtol,
#                 )
#                 result.append(r)
#             return coerce_to_array(result)
#
#         def rhs(x, y):
#             return self.c(x, interpolated_y(x)) + (self.d(x) * integral(x))
#
#         return self._solve_ode(rhs)
#
#     def _interpolate_y(self, y: ndarray) -> inter.interp1d:
#         """
#         Interpolate `y` along `x`, using `interpolation_kind`.
#
#         Parameters
#         ----------
#         y : :class:`numpy.ndarray`
#             The y values to interpolate (probably a guess at the solution).
#
#         Returns
#         -------
#         interpolator : :class:`scipy.interpolate.interp1d`
#             The interpolator function.
#         """
#         return inter.interp1d(
#             x=self.x,
#             y=y,
#             kind=self.interpolation_kind,
#             fill_value="extrapolate",
#             assume_sorted=True,
#         )
#
#     def _solve_ode(self, rhs: Callable) -> ndarray:
#         """Solves an ODE with the given right-hand side."""
#         sol = integ.solve_ivp(
#             fun=rhs,
#             y0=self.y_0,
#             t_span=(self.x[0], self.x[-1]),
#             t_eval=self.x,
#             method=self.ode_method,
#             atol=self.ode_atol,
#             rtol=self.ode_rtol,
#         )
#
#         if not sol.success:
#             raise exceptions.ODESolutionFailed(f"Error while trying to solve ODE: {sol.status}")
#
#         return sol.y
