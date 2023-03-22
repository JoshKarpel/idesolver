import multiprocessing

from numpy import float_, linspace, log
from numpy.typing import NDArray

from idesolver import IDESolver


def run(solver):
    solver.solve()

    return solver


def c(x: NDArray[float_], y: NDArray[float_]) -> NDArray[float_]:
    return y - (0.5 * x) + (1 / (1 + x)) - log(1 + x)


def d(x: NDArray[float_]) -> NDArray[float_]:
    return 1 / (log(2)) ** 2


def k(x: NDArray[float_], s: float) -> NDArray[float_]:
    return x / (1 + s)


def lower_bound(x: NDArray[float_]) -> float:
    return 0


def upper_bound(x: NDArray[float_]) -> float:
    return 1


def f(y: NDArray[float_]) -> NDArray[float_]:
    return y


if __name__ == "__main__":
    # create 20 IDESolvers
    ides = [
        IDESolver(
            x=linspace(0, 1, 100),
            y_0=0,
            c=c,
            d=d,
            k=k,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            f=f,
        )
        for y_0 in linspace(0, 1, 20)
    ]

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run, ides)

    print(results)
