import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from idesolver import IDESolver

FIGS_DIR = Path(__file__).resolve().parent / "assets"

EXTENSIONS = ["png"]


def savefig(name):
    for ext in EXTENSIONS:
        plt.savefig(os.path.join(FIGS_DIR, f"{name}.{ext}"))


def make_comparison_plot(name, solver, exact):
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(solver.x, solver.y, label="IDESolver Solution", linestyle="-", linewidth=3)
    ax.plot(solver.x, exact, label="Analytic Solution", linestyle=":", linewidth=3)

    ax.legend(loc="best")
    ax.grid(True)

    ax.set_title(f"Solution for Global Error Tolerance = {solver.global_error_tolerance}")
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y(x)$", fontsize=12)

    savefig(name)


def make_error_plot(name, solver, exact):
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error, linewidth=3)

    ax.set_yscale("log")
    ax.grid(True)

    ax.set_title(f"Local Error for Global Error Tolerance = {solver.global_error_tolerance}")
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(
        r"$\left| y_{\mathrm{idesolver}}(x) - y_{\mathrm{analytic}}(x) \right|$",
        fontsize=12,
    )

    savefig(name)


def quickstart_example():
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

    make_comparison_plot("quickstart_comparison", solver, exact)
    make_error_plot("quickstart_error", solver, exact)


if __name__ == "__main__":
    print(f"Sending figures to {FIGS_DIR}")
    os.makedirs(FIGS_DIR, exist_ok=True)

    quickstart_example()
