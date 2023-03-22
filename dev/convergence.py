import os
from typing import Tuple

import matplotlib.pyplot as plt
from numpy import abs, cos, float_, imag, linspace, pi, real, sin
from numpy.typing import NDArray

from idesolver import IDESolver

OUT_DIR = os.path.join(os.getcwd(), "out")


def make_comparison_plot(name: str, solver: IDESolver, exact: NDArray[float_]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for iteration, y in solver.y_intermediate.items():
        ax.plot(
            solver.x,
            real(y),
            linestyle="-",
            color="black",
            alpha=0.5 + iteration / solver.iteration,
        )
        ax.plot(
            solver.x,
            imag(y),
            linestyle="--",
            color="black",
            alpha=0.5 + iteration / solver.iteration,
        )

    ax.plot(solver.x, solver._initial_y(), linestyle="-", color="C1")
    ax.plot(solver.x, solver._initial_y(), linestyle="--", color="C1")

    ax.plot(solver.x, real(exact), linestyle="-", color="C0")
    ax.plot(solver.x, imag(exact), linestyle="--", color="C0")

    ax.legend(loc="best")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex_{name}_comparison"))


def make_error_plot(name: str, solver: IDESolver, exact: NDArray[float_]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale("log")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex_{name}_error"))


def example_1() -> Tuple[IDESolver, NDArray[float_]]:
    solver = IDESolver(
        x=linspace(0, 1, 100),
        y_0=1,
        c=lambda x, y: y - cos(2 * pi * x) - (2 * pi * sin(2 * pi * x)) - (0.5 * sin(4 * pi * x)),
        d=lambda x: 1,
        k=lambda x, s: sin(2 * pi * ((2 * x) + s)),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
    )
    solver.solve()
    exact = cos(2 * pi * solver.x)

    return solver, exact


if __name__ == "__main__":
    try:
        os.mkdir(OUT_DIR)
    except FileExistsError:
        pass

    solver, exact = example_1()

    make_comparison_plot("c1", solver, exact)
