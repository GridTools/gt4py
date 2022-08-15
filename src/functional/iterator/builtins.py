from functional.iterator.dispatcher import Dispatcher


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
def cartesian_domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def unstructured_domain(*args):
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
def less_equal(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater_equal(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def not_eq(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_tuple(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tuple_get(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def abs(*args):  # noqa: A001
    raise BackendNotSelectedError()


@builtin_dispatch
def sin(*args):
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
def floor(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def ceil(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def trunc(*args):
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
def minimum(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def maximum(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def fmod(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def power(*args):
    raise BackendNotSelectedError()


UNARY_MATH_NUMBER_BUILTINS = {"abs"}
UNARY_MATH_FP_BUILTINS = {
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
    "floor",
    "ceil",
    "trunc",
}
UNARY_MATH_FP_PREDICATE_BUILTINS = {"isfinite", "isinf", "isnan"}
BINARY_MATH_NUMBER_BUILTINS = {"minimum", "maximum", "fmod", "power"}
MATH_BUILTINS = (
    UNARY_MATH_NUMBER_BUILTINS
    | UNARY_MATH_FP_BUILTINS
    | UNARY_MATH_FP_PREDICATE_BUILTINS
    | BINARY_MATH_NUMBER_BUILTINS
)
BUILTINS = {
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
    "greater",
    "less",
    "less_equal",
    "greater_equal",
    "eq",
    "not_eq",
    "not_",
    "and_",
    "or_",
    "scan",
    "cartesian_domain",
    "unstructured_domain",
    "named_range",
    *MATH_BUILTINS,
}

__all__ = [*BUILTINS]
