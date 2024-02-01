import dataclasses


class Type:
    ...


class Trait:
    ...


@dataclasses.dataclass(frozen=True)
class FunctionParameter:
    """Represents a function parameter within callable types."""

    ty: Type
    """The type of the function parameter."""

    name: str
    """The name of the function parameter."""

    positional: bool
    """
    Whether the corresponding argument can be supplied as a positional in a
    function call.
    """

    keyword: bool
    """
    Whether the corresponding argument can be supplied as a keyword argument
    in a function call.
    """


@dataclasses.dataclass(frozen=True)
class FunctionArgument:
    """Represents an argument to a function call."""

    ty: Type
    """The type of the function call argument."""

    location: int | str
    """
    The position of keyword of the function argument.

    For positional arguments, location is an integer equal to the parameter's
    index. For keyword arguments, location is a string equal to the parameter's
    name.
    """

