# IDESolver

IDESolver is a package that provides an interface for solving
real- or complex-valued integro-differential equations (IDEs) of the form

$$
\begin{aligned}
    \frac{dy}{dx} & = c(y, x) + d(x) \int_{\alpha(x)}^{\beta(x)} k(x, s) \, F( y(s) ) \, ds, \\
    & x \in [a, b], \quad y(a) = y_0.
\end{aligned}
$$

[Integro-differential equations](https://en.wikipedia.org/wiki/Integro-differential_equation) appear in many contexts,
particularly when trying to describe a system whose future behavior depends on its own history and not just its present state.
The IDESolver is an iterative solver,
which means it generates successive approximations to the exact solution,
using each approximation to generate the next (hopefully better) one.
The algorithm is based on a scheme devised by
[Gelmi and Jorquera](https://doi.org/10.1016/j.cpc.2013.09.008>).

If you use IDESolver in your work,
please consider [citing it](https://doi.org/10.21105/joss.00542) [![DOI](https://joss.theoj.org/papers/10.21105/joss.00542/status.svg)](https://doi.org/10.21105/joss.00542).
