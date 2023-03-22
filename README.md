idesolver
---------

[![DOI](https://joss.theoj.org/papers/10.21105/joss.00542/status.svg)](https://doi.org/10.21105/joss.00542)

[![PyPI](https://img.shields.io/pypi/v/idesolver)](https://pypi.org/project/idesolver)
[![PyPI - License](https://img.shields.io/pypi/l/idesolver)](https://pypi.org/project/idesolver)
[![Docs](https://img.shields.io/badge/docs-exist-brightgreen)](https://www.idesolver.how)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/JoshKarpel/idesolver/main.svg)](https://results.pre-commit.ci/latest/github/JoshKarpel/idesolver/main)
[![codecov](https://codecov.io/gh/JoshKarpel/idesolver/branch/main/graph/badge.svg?token=2sjP4V0AfY)](https://codecov.io/gh/JoshKarpel/idesolver)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![GitHub issues](https://img.shields.io/github/issues/JoshKarpel/idesolver)](https://github.com/JoshKarpel/idesolver/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/JoshKarpel/idesolver)](https://github.com/JoshKarpel/idesolver/pulls)

A general purpose numeric integro-differential equation (IDE) solver, based on an iterative scheme devised by [Gelmi and Jorquera](https://doi.org/10.1016/j.cpc.2013.09.008).
IDEs appear in many contexts, particularly when trying to describe a system whose current behavior depends on its own history.
IDESolver provides a simple interface for solving these kinds of equations in Python.

Stable releases are available on PyPI: `pip install idesolver`.

Full documentation can be found [here](https://idesolver.readthedocs.io/en/latest/).
If you use `idesolver` in your research, please consider [citing the associated paper](https://joss.theoj.org/papers/10.21105/joss.00542>).

Problems with IDESolver should be reported via [GitHub issues](https://github.com/JoshKarpel/idesolver/issues).
We are open to improvements: see the [Code of Conduct](https://github.com/JoshKarpel/idesolver/blob/master/CODE_OF_CONDUCT.md)
and the [Contribution Guidelines](https://github.com/JoshKarpel/idesolver/blob/master/CONTRIBUTING.md) for details.
