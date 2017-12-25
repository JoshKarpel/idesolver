import pytest

import numpy as np

from idesolver import IDESolver, IDEConvergenceWarning


def test_warning_when_not_enough_iterations():
    args = dict(
        x = np.linspace(0, 1, 100),
        y_0 = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
        global_error_tolerance = 1e-6,
    )

    good_solver = IDESolver(
        **args
    )
    good_solver.solve()

    bad_solver = IDESolver(
        **args,
        max_iterations = int(good_solver.iteration / 2)
    )

    with pytest.warns(IDEConvergenceWarning):
        bad_solver.solve()


def test_y_intermediate_list_exists_if_store_intermediate_y_is_true():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
        global_error_tolerance = 1e-6,
        store_intermediate_y = True,
    )

    assert hasattr(solver, 'y_intermediate')


def test_number_of_intermediate_solutions_is_same_as_iteration_count_plus_one():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
        global_error_tolerance = 1e-6,
        store_intermediate_y = True,
    )
    solver.solve()

    # the +1 is for the initial value, which isn't counted as an iteration, but is counted as a y_intermediate
    assert len(solver.y_intermediate) == solver.iteration + 1


def test_callback_is_called_correct_number_of_times(mocker):
    callback = mocker.Mock()

    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
        global_error_tolerance = 1e-6,
        store_intermediate_y = True,
    )
    solver.solve(callback = callback)

    # first iteration is number 0, so add one to left to total number of callbacks called
    assert callback.call_count == solver.iteration + 1
