Manual
======

.. currentmodule:: idesolver

The Algorithm
-------------

:class:`IDESolver` implements an iterative algorithm from `this paper <https://doi.org/10.1016/j.cpc.2013.09.008>`_ for solving general IDEs.
The algorithm requires an ODE integrator and a quadrature integrator internally.
IDESolver uses :func:`scipy.integrate.solve_ivp` as the ODE integrator.
The quadrature integrator is either :func:`scipy.integrate.quad` or :func:`complex_quad`, a thin wrapper over :func:`scipy.integrate.quad` which handles splitting the real and imaginary parts of the integral.
