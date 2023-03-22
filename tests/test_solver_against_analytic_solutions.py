from typing import Callable

import pytest
from numpy import allclose, cos, exp, float_, linspace, log, pi, sin, sinh, sqrt
from numpy.typing import NDArray

from idesolver import IDESolver

GELMI_EXAMPLES = [
    (  # 1
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=0,
            c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - log(1 + x),
            d=lambda x: 1 / (log(2)) ** 2,
            k=lambda x, s: x / (1 + s),
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y,
            global_error_tolerance=1e-6,
        ),
        lambda x: log(1 + x),
    ),
    (  # 2
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: y
            - cos(2 * pi * x)
            - (2 * pi * sin(2 * pi * x))
            - (0.5 * sin(4 * pi * x)),
            d=lambda x: 1,
            k=lambda x, s: sin(2 * pi * ((2 * x) + s)),
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y,
            global_error_tolerance=1e-6,
        ),
        lambda x: cos(2 * pi * x),
    ),
    (  # 3
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: 1 - (29 / 60) * x,
            d=lambda x: 1,
            k=lambda x, s: x * s,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y**2,
            global_error_tolerance=1e-6,
        ),
        lambda x: 1 + x + x**2,
    ),
    (  # 4
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: (x * (1 + sqrt(x)) * exp(-sqrt(x))) - (((x**2) + x + 1) * exp(-x)),
            d=lambda x: 1,
            k=lambda x, s: x * s,
            lower_bound=lambda x: x,
            upper_bound=lambda x: sqrt(x),
            f=lambda y: y,
            global_error_tolerance=1e-6,
        ),
        lambda x: exp(-x),
    ),
]

REAL_IDES = [
    (  # RHS = 0
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: 0,
            d=lambda x: 0,
            k=lambda x, s: 0,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: 0,
            global_error_tolerance=1e-6,
        ),
        lambda x: 1,
    ),
    (  # RHS = 0 is the default, so if we pass nothing, we should get that
        IDESolver(x=linspace(0, 1, 100), y_0=1, global_error_tolerance=1e-6),
        lambda x: 1,
    ),
]

COMPLEX_IDES = [
    (
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=0j,
            c=lambda x, y: (5 * y) + 1,
            d=lambda x: -3j,
            k=lambda x, s: 1,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: x,
            f=lambda y: y,
            global_error_tolerance=1e-6,
        ),
        lambda x: 2 * exp(5 * x / 2) * sinh(0.5 * sqrt(25 - 12j) * x) / sqrt(25 - 12j),
    )
]

MULTIDIM = [
    (
        IDESolver(
            x=linspace(0, 7, 100),
            y_0=[0, 1],
            c=lambda x, y: [0.5 * (y[1] + 1), -0.5 * y[0]],
            d=lambda x: -0.5,
            f=lambda y: y,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: x,
        ),
        lambda x: [sin(x), cos(x)],
    )
]


@pytest.mark.parametrize("solver, exact", GELMI_EXAMPLES + REAL_IDES + COMPLEX_IDES + MULTIDIM)
def test_real_ide_against_analytic_solution(
    solver: IDESolver, exact: Callable[[NDArray[float_]], NDArray[float_]]
) -> None:
    solver.solve()

    y_exact = exact(solver.x)

    assert solver.global_error < solver.global_error_tolerance

    assert allclose(solver.y, y_exact, atol=1e-6)
