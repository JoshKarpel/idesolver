import hypothesis as hyp
import hypothesis.strategies as st
import numpy as np
import pytest

from idesolver import IDEConvergenceWarning, IDESolver


def test_warning_when_not_enough_iterations():
    args = dict(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
    )

    good_solver = IDESolver(**args)
    good_solver.solve()

    bad_solver = IDESolver(**args, max_iterations=int(good_solver.iteration / 2))

    with pytest.warns(IDEConvergenceWarning):
        bad_solver.solve()


def test_y_intermediate_list_exists_if_store_intermediate_y_is_true():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
        store_intermediate_y=True,
    )

    assert hasattr(solver, "y_intermediate")


def test_number_of_intermediate_solutions_is_same_as_iteration_count_plus_one():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
        store_intermediate_y=True,
    )
    solver.solve()

    # the +1 is for the initial value, which isn't counted as an iteration, but is counted as a y_intermediate
    assert len(solver.y_intermediate) == solver.iteration + 1


def test_intermediate_solutions_of_scalar_problem_is_list_of_scalar_arrays():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
        store_intermediate_y=True,
    )
    solver.solve()

    assert np.all([y.ndim == 1 for y in solver.y_intermediate])


def test_intermediate_solutions_of_vector_problem_is_list_of_vector_arrays():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=[0, 1, 0],
        c=lambda x, y: [y[0] - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x), y[0], 1],
        d=lambda x: [1 / (np.log(2)) ** 2, 0, 0],
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
        store_intermediate_y=True,
    )
    solver.solve()

    assert np.all([y.shape == (3, 100) for y in solver.y_intermediate])


def test_callback_is_called_correct_number_of_times(mocker):
    callback = mocker.Mock()

    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
        global_error_tolerance=1e-6,
        store_intermediate_y=True,
    )
    solver.solve(callback=callback)

    # first iteration is number 0, so add one to left to get total number of callback calls
    assert callback.call_count == solver.iteration + 1


@pytest.fixture(scope="module")
def default_solver():
    return IDESolver(x=np.linspace(0, 1, 100), y_0=0)


@hyp.given(x=st.complex_numbers(), y=st.complex_numbers())
def test_default_c(default_solver, x, y):
    assert default_solver.c(x, y) == 0


@hyp.given(x=st.complex_numbers())
def test_default_d(default_solver, x):
    assert default_solver.d(x) == 1


@hyp.given(x=st.complex_numbers(), s=st.complex_numbers())
def test_default_k(default_solver, x, s):
    assert default_solver.k(x, s) == 1


@hyp.given(y=st.complex_numbers())
def test_default_f(default_solver, y):
    assert default_solver.f(y) == 0


def test_default_lower_bound(default_solver):
    assert default_solver.lower_bound(default_solver.x) == default_solver.x[0]


def test_default_upper_bound(default_solver):
    assert default_solver.upper_bound(default_solver.x) == default_solver.x[-1]
