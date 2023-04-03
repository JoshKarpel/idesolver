from typing import Tuple

import pytest
from numpy import float_, linspace
from numpy.typing import NDArray


@pytest.fixture(scope="session")
def dummy_args() -> Tuple[NDArray[float_], float]:
    x = linspace(0, 1, 100)
    y_0 = 1

    return x, y_0


x = linspace(0, 1, 100)
y_0 = 1


# @given(max_iterations=integers(min_value=1))
# def test_can_construct_with_positive_max_iterations(max_iterations: int) -> None:
#     IDESolver(x=x, y_0=y_0, max_iterations=max_iterations)
#
#
# @given(max_iterations=integers(max_value=0))
# def test_cannot_construct_with_nonpositive_max_iterations(max_iterations: int) -> None:
#     with pytest.raises(InvalidParameter):
#         IDESolver(x=x, y_0=y_0, max_iterations=max_iterations)
#
#
# @given(smoothing_factor=floats(min_value=0, max_value=1))
# def test_can_construct_with_good_smoothing_factor(smoothing_factor: float) -> None:
#     assume(smoothing_factor != 0 and smoothing_factor != 1)
#
#     IDESolver(x=x, y_0=y_0, smoothing_factor=smoothing_factor)
#
#
# @given(smoothing_factor=one_of(floats(max_value=0), floats(min_value=1)))
# def test_cannot_construct_with_bad_smoothing_factor(smoothing_factor: float) -> None:
#     with pytest.raises(InvalidParameter):
#         IDESolver(x=x, y_0=y_0, smoothing_factor=smoothing_factor)
#
#
# def test_can_construct_with_global_error_tolerance_set_and_without_max_iterations() -> None:
#     IDESolver(x=x, y_0=y_0, global_error_tolerance=1e-6, max_iterations=None)
#
#
# def test_can_construct_with_global_error_tolerance_set_and_with_max_iterations_set() -> None:
#     IDESolver(x=x, y_0=y_0, global_error_tolerance=1e-6, max_iterations=50)
#
#
# def test_can_construct_without_global_error_tolerance_set_and_with_max_iterations() -> None:
#     IDESolver(x=x, y_0=y_0, global_error_tolerance=0, max_iterations=50)
#
#
# def test_cannot_construct_without_global_error_tolerance_set_and_without_max_iterations() -> None:
#     with pytest.raises(InvalidParameter):
#         IDESolver(x=x, y_0=y_0, global_error_tolerance=0, max_iterations=None)
#
#
# @given(global_error_tolerance=floats(min_value=0, exclude_min=True))
# def test_can_construct_with_positive_global_error_tolerance(global_error_tolerance: float) -> None:
#     IDESolver(x=x, y_0=y_0, global_error_tolerance=global_error_tolerance)
#
#
# @given(global_error_tolerance=floats(max_value=0))
# def test_cannot_construct_with_negative_global_error_tolerance(
#     global_error_tolerance: float,
# ) -> None:
#     assume(global_error_tolerance < 0)
#
#     with pytest.raises(InvalidParameter):
#         IDESolver(x=x, y_0=y_0, global_error_tolerance=global_error_tolerance)
