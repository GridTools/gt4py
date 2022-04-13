from functional.iterator.dispatcher import Dispatcher


__all__ = [
    "deref",
    # "can_deref",  # TODO
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
]

builtin_dispatch = Dispatcher()


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


@builtin_dispatch
def deref(*args):
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
