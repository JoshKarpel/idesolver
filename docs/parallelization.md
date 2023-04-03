## Can I pickle an ``IDESolver`` instance?

Yes, with one caveat.
You'll need to define the callables somewhere that Python can find them in the global namespace (i.e., top-level functions in a module, methods in a top-level class, etc.).

## Can I parallelize `IDESolver`?

Not directly: the iterative algorithm is serial by nature.

However, if you have lots of IDEs to solve, you can farm them out to individual cores using Python's `multiprocessing` module (multithreading won't provide any advantage).
Here's an example of using a [`multiprocessing.Pool`][multiprocessing.pool.Pool] to solve several IDEs in parallel:

```python
import multiprocessing
import numpy as np
from idesolver import IDESolver


def run(solver):
    solver.solve()

    return solver


def c(x, y):
    return y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x)


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


if __name__ == "__main__":
    ides = [
        IDESolver(
            x=np.linspace(0, 1, 100),
            y_0=0,
            c=c,
            d=d,
            k=k,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            f=f,
        )
        for y_0 in np.linspace(0, 1, 10)
    ]

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run, ides)

    print(results)
```

Note that the callables all need to defined before the if-name-main so that they can be pickled.
