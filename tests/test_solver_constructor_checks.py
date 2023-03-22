from typing import Tuple

import hypothesis as hyp
import hypothesis.strategies as st
import pytest
from numpy import float_, linspace
from numpy.typing import NDArray

from idesolver import IDESolver, InvalidParameter


@pytest.fixture(scope="session")
def dummy_args() -> Tuple[NDArray[float_], float]:
    x = linspace(0, 1, 100)
    y_0 = 1

    return x, y_0


x = linspace(0, 1, 100)
y_0 = 1


@hyp.given(max_iterations=st.integers(min_value=1))
def test_can_construct_with_positive_max_iterations(max_iterations: int) -> None:
    IDESolver(x=x, y_0=y_0, max_iterations=max_iterations)


@hyp.given(max_iterations=st.integers(max_value=0))
def test_cannot_construct_with_nonpositive_max_iterations(max_iterations: int) -> None:
    with pytest.raises(InvalidParameter):
        IDESolver(x=x, y_0=y_0, max_iterations=max_iterations)


@hyp.given(smoothing_factor=st.floats(min_value=0, max_value=1))
def test_can_construct_with_good_smoothing_factor(smoothing_factor: float) -> None:
    hyp.assume(smoothing_factor != 0 and smoothing_factor != 1)

    IDESolver(x=x, y_0=y_0, smoothing_factor=smoothing_factor)


@hyp.given(smoothing_factor=st.one_of(st.floats(max_value=0), st.floats(min_value=1)))
def test_cannot_construct_with_bad_smoothing_factor(smoothing_factor: float) -> None:
    with pytest.raises(InvalidParameter):
        IDESolver(x=x, y_0=y_0, smoothing_factor=smoothing_factor)


def test_can_construct_with_global_error_tolerance_set_and_without_max_iterations() -> None:
    IDESolver(x=x, y_0=y_0, global_error_tolerance=1e-6, max_iterations=None)


def test_can_construct_with_global_error_tolerance_set_and_with_max_iterations_set() -> None:
    IDESolver(x=x, y_0=y_0, global_error_tolerance=1e-6, max_iterations=50)


def test_can_construct_without_global_error_tolerance_set_and_with_max_iterations() -> None:
    IDESolver(x=x, y_0=y_0, global_error_tolerance=0, max_iterations=50)


def test_cannot_construct_without_global_error_tolerance_set_and_without_max_iterations() -> None:
    with pytest.raises(InvalidParameter):
        IDESolver(x=x, y_0=y_0, global_error_tolerance=0, max_iterations=None)


@hyp.given(global_error_tolerance=st.floats(min_value=0, exclude_min=True))
def test_can_construct_with_positive_global_error_tolerance(global_error_tolerance: float) -> None:
    IDESolver(x=x, y_0=y_0, global_error_tolerance=global_error_tolerance)


@hyp.given(global_error_tolerance=st.floats(max_value=0))
def test_cannot_construct_with_negative_global_error_tolerance(
    global_error_tolerance: float,
) -> None:
    hyp.assume(global_error_tolerance < 0)

    with pytest.raises(InvalidParameter):
        IDESolver(x=x, y_0=y_0, global_error_tolerance=global_error_tolerance)
