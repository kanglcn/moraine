"""The error of wrong arguments: a usage error on the command line (decision 0052)"""

__all__ = ['ArgumentError']


class ArgumentError(ValueError):
    """Arguments of a command or function that its parameters do not accept: a value outside the allowed values or
    range, an argument left out that the others need, arguments that exclude each other, or inputs whose number of
    dimensions, shape, dtype or chunks do not fit the help or each other (e.g. an empty input, or two inputs of
    different numbers of points). The command line reports it as a usage error (exit code 2); problems in the values
    of the data stay ValueErrors (exit code 1)."""
