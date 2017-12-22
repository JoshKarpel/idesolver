import pytest

import numpy as np

from idesolver import IDESolver

GELMI_EXAMPLES = [
    (  # 1
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 0,
            c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
            d = lambda x: 1 / (np.log(2)) ** 2,
            k = lambda x, s: x / (1 + s),
            lower_bound = lambda x: 0,
            upper_bound = lambda x: 1,
            f = lambda y: y,
        ),
        lambda x: np.log(1 + x)
    ),
    (  # 2
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 1,
            c = lambda x, y: y - np.cos(2 * np.pi * x) - (2 * np.pi * np.sin(2 * np.pi * x)) - (.5 * np.sin(4 * np.pi * x)),
            d = lambda x: 1,
            k = lambda x, s: np.sin(2 * np.pi * ((2 * x) + s)),
            lower_bound = lambda x: 0,
            upper_bound = lambda x: 1,
            f = lambda y: y,
        ),
        lambda x: np.cos(2 * np.pi * x)
    ),
    (  # 3
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 1,
            c = lambda x, y: 1 - (29 / 60) * x,
            d = lambda x: 1,
            k = lambda x, s: x * s,
            lower_bound = lambda x: 0,
            upper_bound = lambda x: 1,
            f = lambda y: y ** 2,
        ),
        lambda x: 1 + x + x ** 2
    ),
    (  # 4
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 1,
            c = lambda x, y: (x * (1 + np.sqrt(x)) * np.exp(-np.sqrt(x))) - (((x ** 2) + x + 1) * np.exp(-x)),
            d = lambda x: 1,
            k = lambda x, s: x * s,
            lower_bound = lambda x: x,
            upper_bound = lambda x: np.sqrt(x),
            f = lambda y: y,
        ),
        lambda x: np.exp(-x)
    ),
]


@pytest.mark.parametrize(
    'solver, exact',
    GELMI_EXAMPLES,
    ids = [str(i + 1) for i in range(len(GELMI_EXAMPLES))],
)
def test_Gelmi2014_example(solver, exact):
    solver.solve()

    y_exact = exact(solver.x)

    assert solver.global_error < solver.global_error_tolerance
    assert all(np.abs(solver.y - y_exact) <= 1e-6)
