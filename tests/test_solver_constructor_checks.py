import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from idesolver import IDESolver, InvalidParameter


@hyp.given(max_iterations=st.integers(min_value=1))
def test_can_construct_with_positive_max_iterations(dummy_args, max_iterations):
    IDESolver(*dummy_args, max_iterations=max_iterations)


@hyp.given(max_iterations=st.integers(max_value=0))
def test_cannot_construct_with_nonpositive_max_iterations(dummy_args, max_iterations):
    with pytest.raises(InvalidParameter):
        IDESolver(*dummy_args, max_iterations=max_iterations)


@hyp.given(smoothing_factor=st.floats(min_value=0, max_value=1))
def test_can_construct_with_good_smoothing_factor(dummy_args, smoothing_factor):
    hyp.assume(smoothing_factor != 0 and smoothing_factor != 1)

    IDESolver(*dummy_args, smoothing_factor=smoothing_factor)


@hyp.given(smoothing_factor=st.one_of(st.floats(max_value=0), st.floats(min_value=1)))
def test_cannot_construct_with_bad_smoothing_factor(dummy_args, smoothing_factor):
    with pytest.raises(InvalidParameter):
        IDESolver(*dummy_args, smoothing_factor=smoothing_factor)


def test_can_construct_with_global_error_tolerance_set_and_without_max_iterations(
    dummy_args,
):
    IDESolver(*dummy_args, global_error_tolerance=1e-6, max_iterations=None)


def test_can_construct_with_global_error_tolerance_set_and_with_max_iterations_set(
    dummy_args,
):
    IDESolver(*dummy_args, global_error_tolerance=1e-6, max_iterations=50)


def test_can_construct_without_global_error_tolerance_set_and_with_max_iterations(
    dummy_args,
):
    IDESolver(*dummy_args, global_error_tolerance=0, max_iterations=50)


def test_cannot_construct_without_global_error_tolerance_set_and_without_max_iterations(
    dummy_args,
):
    with pytest.raises(InvalidParameter):
        IDESolver(*dummy_args, global_error_tolerance=0, max_iterations=None)


@hyp.given(global_error_tolerance=st.floats(min_value=0))
def test_can_construct_with_positive_global_error_tolerance(dummy_args, global_error_tolerance):
    hyp.assume(
        global_error_tolerance > 0
    )  # this test does not cover the case where tol = 0 and max_iterations = None

    IDESolver(*dummy_args, global_error_tolerance=global_error_tolerance)


@hyp.given(global_error_tolerance=st.floats(max_value=0))
def test_cannot_construct_with_negative_global_error_tolerance(dummy_args, global_error_tolerance):
    hyp.assume(global_error_tolerance < 0)

    with pytest.raises(InvalidParameter):
        IDESolver(*dummy_args, global_error_tolerance=global_error_tolerance)
