import multiprocessing

import numpy as np

from idesolver import IDESolver


def run(solver):
    solver.solve()

    return solver


def c(x, y):
    return y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x)


def d(x):
    return 1 / (np.log(2)) ** 2


def k(x, s):
    return x / (1 + s)


def lower_bound(x):
    return 0


def upper_bound(x):
    return 1


def f(y):
    return y


if __name__ == '__main__':
    # create 20 IDESolvers
    ides = [
        IDESolver(
            x = np.linspace(0, 1, 100),
            y_0 = 0,
            c = c,
            d = d,
            k = k,
            lower_bound = lower_bound,
            upper_bound = upper_bound,
            f = f,
        )
        for y_0 in np.linspace(0, 1, 20)
    ]

    with multiprocessing.Pool(processes = 2) as pool:
        results = pool.map(run, ides)

    print(results)
