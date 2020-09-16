import numpy as np
import pytest

from idesolver import IDESolver, UnexpectedlyComplexValuedIDE


def test_raise_exception_if_unexpectedly_complex():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,  # this not being 0j is what makes the test fail
        c=lambda x, y: (5 * y) + 1,
        d=lambda x: -3j,
        k=lambda x, s: 1,
        lower_bound=lambda x: 0,
        upper_bound=lambda x: x,
        f=lambda y: y,
    )

    with pytest.raises(UnexpectedlyComplexValuedIDE):
        solver.solve()


def test_no_exception_if_expected_complex():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0j,
        c=lambda x, y: (5 * y) + 1,
        d=lambda x: -3j,
        k=lambda x, s: 1,
        lower_bound=lambda x: 0,
        upper_bound=lambda x: x,
        f=lambda y: y,
    )

    solver.solve()
