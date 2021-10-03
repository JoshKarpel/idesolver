Change Log
==========

.. currentmodule:: idesolver

v1.2.0
------

* :math:`k(x, s)` can now be a matrix.
* `UnexpectedlyComplexValuedIDE` is no longer raised if any `TypeError` occurs during integration.
  This may lead to less-explicit errors, but this is preferable to incorrect error reporting.

v1.1.0
------
* Add support for multidimensional IDEs (PR :pr:`35` resolves :issue:`28`, thanks `nbrucy <https://github.com/nbrucy>`_!)

v1.0.5
------
* Relaxes dependency version restrictions in advance of changes to ``pip``.
  There shouldn't be any impact on users.

v1.0.4
------
* Revision of packaging and CI flow. There shouldn't be any impact on users.

v1.0.3
------
* Revision of package structure and CI flow. There shouldn't be any impact on users.

v1.0.2
------
* IDESolver now explicitly requires Python 3.6+ on install. Dependencies on ``numpy`` and ``scipy`` are given as lower bounds.

v1.0.1
------
* Changed the name of ``IDESolver.F`` to ``f``, as intended.
* The default global error function is now injected instead of hard-coded.

v1.0.0
------
Initial release.
