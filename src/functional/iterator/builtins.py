from functional.iterator.dispatcher import Dispatcher


__all__ = [
    "deref",
    "can_deref",
    "shift",
    "lift",
    "reduce",
    "plus",
    "minus",
    "multiplies",
    "divides",
    "make_tuple",
    "tuple_get",
    "if_",
    "greater",  # TODO not in c++
    "less",
    "eq",
    "not_",
    "and_",
    "or_",
    "scan",
    "domain",
    "named_range",
    "abs",
    "min",
    "max",
    "mod",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sqrt",
    "exp",
    "log",
    "gamma",
    "cbrt",
    "isfinite",
    "isinf",
    "isnan",
    "floor",
    "ceil",
    "trunc",
]

builtin_dispatch = Dispatcher()


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


@builtin_dispatch
def deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def can_deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def shift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def lift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def reduce(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def scan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def named_range(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def if_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def not_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def and_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def or_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def minus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def plus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def multiplies(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def divides(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def eq(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def less(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_tuple(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tuple_get(*args):
    raise BackendNotSelectedError()


# FIXME(ben): various of the following built-ins will shadow python built-ins!
# We should find a good way around this...

@builtin_dispatch
def abs(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def min(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def max(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def mod(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def sin (*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cos(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arcsin(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arccos(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arctan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def sinh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cosh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tanh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arcsinh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arccosh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arctanh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def sqrt(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def exp(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def log(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def gamma(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cbrt(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isfinite(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isinf(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isnan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def floor(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def ceil(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def trunc(*args):
    raise BackendNotSelectedError()
