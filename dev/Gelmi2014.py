import time

import numpy as np
import scipy as sp
import scipy.integrate as integ
import scipy.optimize as optim
import scipy.interpolate as inter

import matplotlib.pyplot as plt

from idesolver import IDESolver


def make_comparison_plot(solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lines = [solver.initial_guess(), solver.y, exact]
    labels = ['initial guess', 'solution', 'exact']
    styles = ['-', '-', '--']

    for y, label, style in zip(lines, labels, styles):
        ax.plot(solver.x, y, label = label, linestyle = style)

    ax.legend(loc = 'best')

    plt.show()


def make_error_plot(solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale('log')
    ax.grid(True)

    plt.show()


def example_1():
    solver = IDESolver(
        y_initial = 0,
        x = np.linspace(0, 1, 100),
        c = lambda y, x: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
    )
    solver.solve()
    exact = np.log(1 + solver.x)

    print(solver.iterations)

    make_comparison_plot(solver, exact)
    make_error_plot(solver, exact)


def example_2():
    solver = IDESolver(
        y_initial = 1,
        x = np.linspace(0, 1, 100),
        c = lambda y, x: y - np.cos(2 * np.pi * x) - (2 * np.pi * np.sin(2 * np.pi * x)) - (.5 * np.sin(4 * np.pi * x)),
        d = lambda x: 1,
        k = lambda x, s: np.sin(2 * np.pi * ((2 * x) + s)),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
    )
    solver.solve()
    exact = np.cos(2 * np.pi * solver.x)

    print(solver.iterations)

    make_comparison_plot(solver, exact)
    make_error_plot(solver, exact)


def example_3():
    solver = IDESolver(
        y_initial = 1,
        x = np.linspace(0, 1, 100),
        c = lambda y, x: 1 - (29 / 60) * x,
        d = lambda x: 1,
        k = lambda x, s: x * s,
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y ** 2,
    )
    solver.solve()
    exact = 1 + solver.x + solver.x ** 2

    print(solver.iterations)

    make_comparison_plot(solver, exact)
    make_error_plot(solver, exact)


if __name__ == '__main__':
    example_2()
