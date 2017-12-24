import pytest

import numpy as np

from idesolver import IDESolver, UnexpectedlyComplexValuedIDE


def test_raise_exception_if_unexpectedly_complex():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 0,  # this not being 0j is what makes the test fail
        c = lambda x, y: (5 * y) + 1,
        d = lambda x: -3j,
        k = lambda x, s: 1,
        lower_bound = lambda x: 0,
        upper_bound = lambda x: x,
        f = lambda y: y,
    )

    with pytest.raises(UnexpectedlyComplexValuedIDE):
        solver.solve()


COMPLEX_IDES = [
    (  # 1
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 0j,
            c = lambda x, y: (5 * y) + 1,
            d = lambda x: -3j,
            k = lambda x, s: 1,
            lower_bound = lambda x: 0,
            upper_bound = lambda x: x,
            f = lambda y: y,
        ),
        lambda x: 2 * np.exp(5 * x / 2) * np.sinh(.5 * np.sqrt(25 - 12j) * x) / np.sqrt(25 - 12j)
    ),
]


@pytest.mark.parametrize(
    'solver, exact',
    COMPLEX_IDES,
    ids = [str(i + 1) for i in range(len(COMPLEX_IDES))],
)
def test_complex_ides_with_analytic_solutions(solver, exact):
    solver.solve()

    y_exact = exact(solver.x)

    assert solver.global_error < solver.global_error_tolerance

    # the following is NOT strictly true
    # it happens to be true for the parameters above, so it's a good smoke test when changing the algorithm
    assert np.allclose(solver.y, y_exact, atol = 1e-6)
