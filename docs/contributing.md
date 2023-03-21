# Contributing Guide

!!! info "IDESolver is open to contributions!"

    - [Report bugs and request features](https://github.com/JoshKarpel/idesolver/issues)
    - [General discussion](https://github.com/JoshKarpel/idesolver/discussions)
    - [Pull requests](https://github.com/JoshKarpel/idesolver/pulls)

## Development Environment

IDESolver uses:

- [`poetry`](https://python-poetry.org) to manage development dependencies.
- [`pre-commit`](https://pre-commit.com) to run various linters and formatters.
- [`pytest`](https://docs.pytest.org) for testing and [`mypy`](https://mypy-lang.org) for static type-checking.
- [`mkdocs`](https://www.mkdocs.org) with the [Material theme](https://squidfunk.github.io/mkdocs-material) for documentation.

### Initial Setup

To set up a local development environment after cloning the repository:

1. [Install `poetry`](https://python-poetry.org/docs/#installation).
2. Run `poetry shell` to create a virtual environment for `idesolver` and spawn a new shell session with that virtual environment activated.
   In the future you'll run `poetry shell` again to activate the virtual environment.
3. Run `poetry install` to install IDESolver's dependencies.
4. Run `pre-commit install` to configure `pre-commit`'s integration with `git`.
   Do not commit without `pre-commit` installed!

### Running Tests and Type-Checking

Run `pytest` to run tests.

Run `mypy` to check types.

### Building the Docs Locally

To build the docs and start a local web server to view the results of your edits with live reloading, run
```bash
mkdocs serve
```
from the repository root.
