import hypothesis as hyp
import hypothesis.strategies as st
import numpy as np
import pytest
from pytest_mock import MockerFixture

from idesolver import IDE, IDEConvergenceWarning, solve_ide


def test_warning_when_not_enough_iterations() -> None:
    ide = IDE(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
    )
    good_result = solve_ide(ide=ide)

    with pytest.warns(IDEConvergenceWarning):
        bad_result = solve_ide(
            ide=ide,
            max_iterations=int(good_result.iterations / 2),
        )


def test_callback_is_called_correct_number_of_times(mocker: MockerFixture) -> None:
    callback = mocker.Mock()

    result = solve_ide(
        ide=IDE(
            x=np.linspace(0, 1, 100),
            y_0=0,
            c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
            d=lambda x: 1 / (np.log(2)) ** 2,
            k=lambda x, s: x / (1 + s),
            lower_bound=lambda x: 0,
            upper_bound=lambda x: 1,
            f=lambda y: y,
        ),
        global_error_tolerance=1e-6,
        callback=callback,
    )

    # first iteration is number 0, so add one to left to get total number of callback calls
    assert callback.call_count == result.iterations + 1


@pytest.fixture(scope="module")
def default_solver() -> IDE:
    return IDE(x=np.linspace(0, 1, 100), y_0=0)


@hyp.given(x=st.complex_numbers(), y=st.complex_numbers())
def test_default_c(default_solver: IDE, x: complex, y: complex) -> None:
    assert default_solver.c(x, y) == 0


@hyp.given(x=st.complex_numbers())
def test_default_d(default_solver: IDE, x: complex) -> None:
    assert default_solver.d(x) == 1


@hyp.given(x=st.complex_numbers(), s=st.complex_numbers())
def test_default_k(default_solver: IDE, x: complex, s: complex) -> None:
    assert default_solver.k(x, s) == 1


@hyp.given(y=st.complex_numbers())
def test_default_f(default_solver: IDE, y: complex) -> None:
    assert default_solver.f(y) == 0


def test_default_lower_bound(default_solver: IDE) -> None:
    assert default_solver.lower_bound(default_solver.x) == default_solver.x[0]


def test_default_upper_bound(default_solver: IDE) -> None:
    assert default_solver.upper_bound(default_solver.x) == default_solver.x[-1]
