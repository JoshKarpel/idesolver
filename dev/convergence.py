import os

import numpy as np
import matplotlib.pyplot as plt

from idesolver import IDESolver

OUT_DIR = os.path.join(os.getcwd(), "out")


def make_comparison_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for iteration, y in solver.y_intermediate.items():
        ax.plot(
            solver.x,
            np.real(y),
            linestyle="-",
            color="black",
            alpha=0.5 + iteration / solver.iteration,
        )
        ax.plot(
            solver.x,
            np.imag(y),
            linestyle="--",
            color="black",
            alpha=0.5 + iteration / solver.iteration,
        )

    ax.plot(solver.x, solver._initial_y(), linestyle="-", color="C1")
    ax.plot(solver.x, solver._initial_y(), linestyle="--", color="C1")

    ax.plot(solver.x, np.real(exact), linestyle="-", color="C0")
    ax.plot(solver.x, np.imag(exact), linestyle="--", color="C0")

    ax.legend(loc="best")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex_{name}_comparison"))


def make_error_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale("log")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex_{name}_error"))


def example_1():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=1,
        c=lambda x, y: y
        - np.cos(2 * np.pi * x)
        - (2 * np.pi * np.sin(2 * np.pi * x))
        - (0.5 * np.sin(4 * np.pi * x)),
        d=lambda x: 1,
        k=lambda x, s: np.sin(2 * np.pi * ((2 * x) + s)),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
    )
    solver.solve()
    exact = np.cos(2 * np.pi * solver.x)

    return solver, exact


if __name__ == "__main__":
    try:
        os.mkdir(OUT_DIR)
    except FileExistsError:
        pass

    solver, exact = example_1()

    make_comparison_plot("c1", solver, exact)
