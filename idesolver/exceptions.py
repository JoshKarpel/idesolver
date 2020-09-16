class IDEConvergenceWarning(Warning):
    """The solver is not converging."""

    pass


class IDESolverException(Exception):
    """Base exception for all IDESolver exceptions."""

    pass


class InvalidParameter(IDESolverException):
    """Invalid parameters were passed to the solver's constructor."""

    pass


class ODESolutionFailed(IDESolverException):
    """IDESolver was not able to find a solution."""

    pass


class UnexpectedlyComplexValuedIDE(IDESolverException):
    """
    IDESolver was expecting a real-valued IDE based on the inputs, but it
    appears to be complex-valued.
    """

    pass
