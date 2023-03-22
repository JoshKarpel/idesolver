from typing import Callable

import pytest
import scipy.integrate as integ
from numpy import exp, float_, linspace, log
from numpy.typing import NDArray

from idesolver import complex_quad


@pytest.fixture(scope="module")
def x() -> NDArray[float]:
    return linspace(0, 1, 1000)


ArrayToArray = Callable[[NDArray[float_]], NDArray[float_]]


@pytest.fixture(
    scope="module",
    params=[
        lambda x: 1,
        lambda x: 2.3 * x,
        lambda x: 0.1 * x**2,
        lambda x: exp(x),
        lambda x: log(x),
        lambda x: 1 / (x + 0.1),
    ],
)
def real_integrand(request) -> ArrayToArray:
    return request.param


def test_real_part_passes_through(x: NDArray[float], real_integrand: ArrayToArray) -> None:
    cq_result, cq_real_error, cq_imag_error, *_ = complex_quad(real_integrand, x[0], x[-1])

    quad_result, quad_error = integ.quad(real_integrand, x[0], x[-1])

    assert cq_result == quad_result
    assert cq_real_error == quad_error


def test_imag_part_passes_through(x: NDArray[float], real_integrand: ArrayToArray) -> None:
    imag_integrand = lambda x: 1j * real_integrand(x)

    cq_result, cq_real_error, cq_imag_error, *_ = complex_quad(imag_integrand, x[0], x[-1])

    quad_result, quad_error = integ.quad(real_integrand, x[0], x[-1])

    assert cq_result == 1j * quad_result
    assert cq_imag_error == quad_error


# copy real_integrand so we can use the fixture twice
second_integrand = real_integrand


def test_real_and_imag_parts_combined(
    x: NDArray[float], real_integrand: ArrayToArray, second_integrand: ArrayToArray
) -> None:
    imag_integrand = lambda x: 1j * second_integrand(x)
    combined_integrand = lambda x: real_integrand(x) + imag_integrand(x)

    cq_result, cq_real_error, cq_imag_error, *_ = complex_quad(combined_integrand, x[0], x[-1])

    quad_real_result, quad_real_error = integ.quad(real_integrand, x[0], x[-1])
    quad_imag_result, quad_imag_error = integ.quad(second_integrand, x[0], x[-1])

    assert cq_result == quad_real_result + (1j * quad_imag_result)
    assert cq_real_error == quad_real_error
    assert cq_imag_error == quad_imag_error
