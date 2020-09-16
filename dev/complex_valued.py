import os

import matplotlib.pyplot as plt
import numpy as np

from idesolver import IDESolver

OUT_DIR = os.path.join(os.getcwd(), "out", __file__.strip(".py"))


def make_comparison_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = [solver.y, exact]
    labels = ["solution", "exact"]
    colors = ["C0", "C1"]

    for y, label, color in zip(lines, labels, colors):
        ax.plot(solver.x, np.real(y), label=r"R " + label, color=color, linestyle="-")
        ax.plot(solver.x, np.imag(y), label=r"I " + label, color=color, linestyle="--")

    ax.legend(loc="best")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex{name}_comparison"))


def make_error_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale("log")
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f"ex{name}_error"))


def example_1():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0j,
        c=lambda x, y: (5 * y) + 1,
        d=lambda x: -3j,
        k=lambda x, s: 1,
        lower_bound=lambda x: 0,
        upper_bound=lambda x: x,
        f=lambda y: y,
    )
    solver.solve()
    exact = (
        2
        * np.exp(5 * solver.x / 2)
        * np.sinh(0.5 * np.sqrt(25 - 12j) * solver.x)
        / np.sqrt(25 - 12j)
    )

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
