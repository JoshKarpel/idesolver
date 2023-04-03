from typing import Callable

import pytest
from numpy import allclose, cos, exp, float_, linspace, log, pi, sin, sinh, sqrt
from numpy.typing import NDArray

from idesolver import IDE, solve_ide

GELMI_EXAMPLES = [
    (  # 1
        IDE(
            x=linspace(0, 1, 100),
            y_0=0,
            c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - log(1 + x),
            d=lambda x: 1 / (log(2)) ** 2,
            k=lambda x, s: x / (1 + s),
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y,
        ),
        lambda x: log(1 + x),
    ),
    (  # 2
        IDE(
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
        ),
        lambda x: cos(2 * pi * x),
    ),
    (  # 3
        IDE(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: 1 - (29 / 60) * x,
            d=lambda x: 1,
            k=lambda x, s: x * s,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y**2,
        ),
        lambda x: 1 + x + x**2,
    ),
    (  # 4
        IDE(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: (x * (1 + sqrt(x)) * exp(-sqrt(x))) - (((x**2) + x + 1) * exp(-x)),
            d=lambda x: 1,
            k=lambda x, s: x * s,
            lower_bound=lambda x: x,
            upper_bound=lambda x: sqrt(x),
            f=lambda y: y,
        ),
        lambda x: exp(-x),
    ),
]

REAL_IDES = [
    (  # RHS = 0
        IDE(
            x=linspace(0, 1, 100),
            y_0=1,
            c=lambda x, y: 0,
            d=lambda x: 0,
            k=lambda x, s: 0,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: 0,
        ),
        lambda x: 1,
    ),
    (  # RHS = 0 is the default, so if we pass nothing, we should get that
        IDE(x=linspace(0, 1, 100), y_0=1),
        lambda x: 1,
    ),
]

COMPLEX_IDES = [
    (
        IDE(
            x=linspace(0, 1, 100),
            y_0=0j,
            c=lambda x, y: (5 * y) + 1,
            d=lambda x: -3j,
            k=lambda x, s: 1,
            lower_bound=lambda x: 0,
            upper_bound=lambda x: x,
            f=lambda y: y,
        ),
        lambda x: 2 * exp(5 * x / 2) * sinh(0.5 * sqrt(25 - 12j) * x) / sqrt(25 - 12j),
    )
]

MULTIDIM = [
    (
        IDE(
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


@pytest.mark.parametrize("ide, exact", GELMI_EXAMPLES + REAL_IDES + COMPLEX_IDES + MULTIDIM)
def test_real_ide_against_analytic_solution(
    ide: IDE, exact: Callable[[NDArray[float_]], NDArray[float_]]
) -> None:
    result = solve_ide(ide)

    y_exact = exact(ide.x)

    assert result.global_error < 1e-6

    assert allclose(result.y, y_exact, atol=1e-6)
