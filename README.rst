idesolver
---------

.. image:: http://joss.theoj.org/papers/9d3ba306da6abb37f7cf357cd9aad695/status.svg
    :target: http://joss.theoj.org/papers/9d3ba306da6abb37f7cf357cd9aad695

.. image:: https://readthedocs.org/projects/idesolver/badge/?version=latest
    :target: https://idesolver.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/v/idesolver
    :alt: PyPI

.. image:: https://codecov.io/gh/JoshKarpel/idesolver/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/JoshKarpel/idesolver

.. image:: https://results.pre-commit.ci/badge/github/JoshKarpel/idesolver/master.svg
    :target: https://results.pre-commit.ci/latest/github/JoshKarpel/idesolver/master
    :alt: pre-commit.ci status

A general purpose numeric integro-differential equation (IDE) solver, based on an iterative scheme devised by `Gelmi and Jorquera <https://doi.org/10.1016/j.cpc.2013.09.008>`_.
IDEs appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
IDESolver provides a simple interface for solving these kinds of equations in Python.

Stable releases are available on PyPI: ``pip install idesolver``.
IDESolver requires Python 3.6+, `numpy <https://pypi.python.org/pypi/numpy>`_, and `scipy <https://pypi.python.org/pypi/scipy/>`_.
We recommend installing into a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.

Full documentation can be found `here <https://idesolver.readthedocs.io/en/latest/>`_.
If you use ``idesolver`` in your research, please consider `citing the associated paper <https://joss.theoj.org/papers/10.21105/joss.00542>`_.

Details about running the test suite are at the end of the `manual <https://idesolver.readthedocs.io/en/latest/manual.html>`_.
Problems with IDESolver should be reported via `GitHub issues <https://github.com/JoshKarpel/idesolver/issues>`_.
We are open to improvements: see the `Code of Conduct <https://github.com/JoshKarpel/idesolver/blob/master/CODE_OF_CONDUCT.md>`_ and the `Contribution Guidelines <https://github.com/JoshKarpel/idesolver/blob/master/CONTRIBUTING.md>`_ for details.
