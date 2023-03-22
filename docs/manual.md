IDESolver implements an iterative algorithm from [this paper](https://doi.org/10.1016/j.cpc.2013.09.008) for solving general IDEs.
The algorithm requires an ODE integrator and a quadrature integrator internally.
IDESolver uses [`scipy.integrate.solve_ivp`][scipy.integrate.solve_ivp] as the ODE integrator.
The quadrature integrator is either [`scipy.integrate.quad`][scipy.integrate.quad] or [`idesolver.complex_quad`][idesolver.complex_quad],
a thin wrapper over [`scipy.integrate.quad`][scipy.integrate.quad] which handles splitting the real and imaginary parts of the integral.

## The Algorithm

We want to find an approximate solution to

$$
\begin{aligned}
    \frac{dy}{dx} & = c(y, x) + d(x) \int_{\alpha(x)}^{\beta(x)} k(x, s) \, F( y(s) ) \, ds, \\
    & x \in [a, b], \quad y(a) = y_0.
\end{aligned}
$$

The algorithm begins by creating an initial guess for $y$ by using an ODE solver on

$$
    \frac{dy}{dx} = c(y, x)
$$

Since there's no integral on the right-hand-side, standard ODE solvers can handle it easily.
Call this guess $y^{(0)}$.
We can then produce a better guess by seeing what we would get with the original IDE, but replacing $y` on the right-hand-side by :math:`y^{(0)}$:

$$
    \frac{dy^{(1/2)}}{dx} = c(y^{(0)}, x) + d(x) \int_{\alpha(x)}^{\beta(x)} k(x, s) \, F( y^{(0)}(s) ) \, ds
$$

Again, this is just an ODE, because $y^{(1/2)}$ does not appear on the right.
At this point in the algorithm we check the global error between $y^{(0)}$ and $y^{(1/2)}$.
If it's smaller than the tolerance, we stop iterating and take $y^{(1/2)}$ to be the solution.
If it's larger than the tolerance, the iteration continues.
To be conservative and to make sure we don't over-correct, we'll combine $y^{(1/2)}$ with $y^{(0)}$.

$$
    y^{(1)} = \alpha \, y^{(0)} + (1 - \alpha) \, y^{(1/2)}
$$

The process then repeats: solve the IDE-turned-ODE with $y^{(1)}$ on the right-hand-side, see how different it is, maybe make a new guess, etc.

## Stopping Conditions

IDESolver can operate in three modes: either a nonzero global error tolerance should be given, or a maximum number of iterations should be given, or both should be given.

- Nonzero global error tolerance is the standard mode, as described above.
- If a maximum number of iterations is given with zero global error tolerance, the algorithm will iterate that many times and then stop.
- If both are given, the algorithm terminates if either condition is met.


## Global Error Estimate

The default global error estimate $G` between two possible solutions $y_1$ and $y_2$ is

$$
    G(y_1, y_2) = \sqrt{ \sum_{x_i} \left| y_1(x_i) - y_2(x_i) \right| }
$$

A different global error estimator can be passed in the constructor as the argument `global_error_function`.
