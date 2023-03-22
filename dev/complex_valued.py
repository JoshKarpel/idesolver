import os
from typing import Tuple

import matplotlib.pyplot as plt
from numpy import abs, complex_, exp, imag, linspace, real, sinh, sqrt
from numpy.typing import NDArray

from idesolver import IDESolver

OUT_DIR = os.path.join(os.getcwd(), "out", __file__.strip(".py"))


def make_comparison_plot(name: str, solver: IDESolver, exact: NDArray[complex_]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = [solver.y, exact]
    labels = ["solution", "exact"]
    colors = ["C0", "C1"]

    for y, label, color in zip(lines, labels, colors):
        ax.plot(solver.x, real(y), label=r"R " + label, color=color, linestyle="-")
        ax.plot(solver.x, imag(y), label=r"I " + label, color=color, linestyle="--")

    ax.legend(loc="best")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex{name}_comparison"))


def make_error_plot(name: str, solver: IDESolver, exact: NDArray[complex_]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale("log")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex{name}_error"))


def example_1() -> Tuple[IDESolver, NDArray[complex_]]:
    solver = IDESolver(
        x=linspace(0, 1, 100),
        y_0=0j,
        c=lambda x, y: (5 * y) + 1,
        d=lambda x: -3j,
        k=lambda x, s: 1,
        lower_bound=lambda x: 0,
        upper_bound=lambda x: x,
        f=lambda y: y,
    )
    solver.solve()
    exact = 2 * exp(5 * solver.x / 2) * sinh(0.5 * sqrt(25 - 12j) * solver.x) / sqrt(25 - 12j)

    # for s, e in zip(solver.y, exact):
    #     print(s, e)

    return solver, exact


if __name__ == "__main__":
    try:
        os.mkdir(OUT_DIR)
    except FileExistsError:
        pass

    solver, exact = example_1()

    make_comparison_plot("c1", solver, exact)
    make_error_plot("c1", solver, exact)
