Overview
========

.. currentmodule:: idesolver

`IDESolver <https://github.com/JoshKarpel/idesolver>`_ is a package that provides an interface for solving real- or complex-valued integro-differential equations (IDEs) of the form

.. math::

   \frac{dy}{dx} & = c(y, x) + d(x) \int_{\alpha(x)}^{\beta(x)} k(x, s) \, F( y(s) ) \, ds, \\
   & x \in [a, b], \quad y(a) = y_0.

Integro-differential equations appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
The IDESolver is an iterative solver, which means it generates successive approximations to the exact solution, using each approximation to generate the next (hopefully better) one.
The algorithm is based on a scheme devised by `Gelmi and Jorquera <https://doi.org/10.1016/j.cpc.2013.09.008>`_.

If you use IDESolver in your work, please consider `citing it <https://doi.org/10.21105/joss.00542>`_.

:doc:`quickstart`
   A brief tutorial in using IDESolver.

:doc:`manual`
   Details about the implementation of IDESolver.
   Includes information about running the test suite.

:doc:`api`
   Detailed documentation for IDESolver's API.

:doc:`faq`
   These are questions are asked, sometimes frequently.

:doc:`changelog`
   Change logs going back to the initial release.


.. toctree::
   :hidden:
   :maxdepth: 2

   self
   quickstart
   manual
   api
   faq
   changelog
