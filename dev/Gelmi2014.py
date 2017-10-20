import os

import numpy as np
import matplotlib.pyplot as plt

from idesolver import IDESolver

OUT_DIR = os.path.join(os.getcwd(), 'out')


def make_comparison_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = [solver.initial_y(), solver.y, exact]
    labels = ['initial guess', 'solution', 'exact']
    styles = ['-', '-', '--']

    for y, label, style in zip(lines, labels, styles):
        ax.plot(solver.x, y, label = label, linestyle = style)

    ax.legend(loc = 'best')
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f'ex{name}_comparison'))


def make_error_plot(name, solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale('log')
    ax.grid(True)

    plt.savefig(os.path.join(OUT_DIR, f'ex{name}_error'))


def example_1():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
    )
    solver.solve()
    exact = np.log(1 + solver.x)

    return solver, exact


def example_2():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 1,
        c = lambda x, y: y - np.cos(2 * np.pi * x) - (2 * np.pi * np.sin(2 * np.pi * x)) - (.5 * np.sin(4 * np.pi * x)),
        d = lambda x: 1,
        k = lambda x, s: np.sin(2 * np.pi * ((2 * x) + s)),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
    )
    solver.solve()
    exact = np.cos(2 * np.pi * solver.x)

    return solver, exact


def example_3():
    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_0 = 1,
        c = lambda x, y: 1 - (29 / 60) * x,
        d = lambda x: 1,
        k = lambda x, s: x * s,
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y ** 2,
    )
    solver.solve()
    exact = 1 + solver.x + solver.x ** 2

    return solver, exact


if __name__ == '__main__':
    try:
        os.mkdir(OUT_DIR)
    except FileExistsError:
        pass

    solver, exact = example_1()
    # solver, exact = example_2()
    # solver, exact = example_3()

    make_comparison_plot('1', solver, exact)
    make_error_plot('1', solver, exact)
