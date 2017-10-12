.. currentmodule:: idesolver

Quickstart
==========

Suppose we want to solve the IDE

.. math::
\frac{dy}{dx} & = y(x) - \frac{x}{2} + \frac{1}{1 + x} - \ln(1 + x) + \frac{1}{\left(\ln(2)\right)^2} \int_0^1 \frac{x}{1 + s} \, y(s) \, ds, \\
    & \mathrm{on} \quad 0 \leq x \leq 1, \quad y(0) = 0.

This analytic solution to this IDE is :math:`y(x) = \ln(1 + x)`.

We begin by creating an instance of :class:`IDESolver`, which contains information about the integro-differential equation (IDE) that we want to solve, as well as options to control the accuracy of the solution.

.. code-block:: python

    import numpy as np

    from idesolver import IDESolver

    solver = IDESolver(
        x = np.linspace(0, 1, 100),
        y_initial = 0,
        c = lambda x, y: y - (.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d = lambda x: 1 / (np.log(2)) ** 2,
        k = lambda x, s: x / (1 + s),
        lower_bound = lambda x: 0,
        upper_bound = lambda x: 1,
        f = lambda y: y,
    )
