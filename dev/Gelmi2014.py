import os

import numpy as np
import matplotlib.pyplot as plt

from idesolver import IDESolver

OUT_DIR = __file__.strip(".py")


def make_comparison_plot(name, solver, exact):
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    lines = [solver._initial_y(), solver.y, exact]
    labels = ["initial guess", "solution", "exact"]
    styles = ["-", "-", "--"]

    for y, label, style in zip(lines, labels, styles):
        ax.plot(solver.x, y, label=label, linestyle=style)

    ax.legend(loc="best")
    ax.grid(True)

    ax.set_title(
        f"Solution for Global Error Tolerance = {solver.global_error_tolerance}"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y(x)$")

    plt.savefig(
        os.path.join(
            OUT_DIR, f"example_{name}_comparison_at_tol={solver.global_error_tolerance}"
        )
    )


def make_error_plot(name, solver, exact):
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale("log")
    ax.grid(True)

    ax.set_title(
        f"Local Error for Global Error Tolerance = {solver.global_error_tolerance}"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\left| y_{\mathrm{solver}}(x) - y_{\mathrm{exact}}(x) \right|$")

    plt.savefig(
        os.path.join(
            OUT_DIR, f"example_{name}_error_at_tol={solver.global_error_tolerance}"
        )
    )


def example_1():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=0,
        c=lambda x, y: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y,
    )
    solver.solve()
    exact = np.log(1 + solver.x)

    return solver, exact


def example_2():
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


def example_3():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=1,
        c=lambda x, y: 1 - (29 / 60) * x,
        d=lambda x: 1,
        k=lambda x, s: x * s,
        lower_bound=lambda x: 0,
        upper_bound=lambda x: 1,
        f=lambda y: y ** 2,
    )
    solver.solve()
    exact = 1 + solver.x + solver.x ** 2

    return solver, exact


def example_4():
    solver = IDESolver(
        x=np.linspace(0, 1, 100),
        y_0=1,
        c=lambda x, y: (x * (1 + np.sqrt(x)) * np.exp(-np.sqrt(x)))
        - (((x ** 2) + x + 1) * np.exp(-x)),
        d=lambda x: 1,
        k=lambda x, s: x * s,
        lower_bound=lambda x: x,
        upper_bound=lambda x: np.sqrt(x),
        f=lambda y: y,
    )
    solver.solve()
    exact = np.exp(-solver.x)

    return solver, exact


if __name__ == "__main__":
    try:
        os.mkdir(OUT_DIR)
    except FileExistsError:
        pass

    print(f"Sending output to {OUT_DIR}")

    examples = [example_1, example_2, example_3, example_4]

    for name, example in enumerate(examples, start=1):
        solver, exact = example()

        print(
            f"Example {name} took {solver.iteration} iterations to get to global error {solver.global_error}. Error compared to analytic solution is {solver._global_error(solver.y, exact)}"
        )

        make_comparison_plot(name, solver, exact)
        make_error_plot(name, solver, exact)
