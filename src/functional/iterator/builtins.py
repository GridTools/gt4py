from functional.iterator.dispatcher import Dispatcher


__all__ = [
    "deref",
    "shift",
    "lift",
    "reduce",
    "scan",
    "is_none",
    "domain",
    "named_range",
    "compose",
    "if_",
    "or_",
    "minus",
    "plus",
    "mul",
    "div",
    "eq",
    "greater",
    "make_tuple",
    "nth",
    "plus",
    "reduce",
    "scan",
    "shift",
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
def is_none(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def named_range(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def compose(sten):
    raise BackendNotSelectedError()


@builtin_dispatch
def if_(*args):
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
def mul(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def div(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def eq(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_tuple(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def nth(*args):
    raise BackendNotSelectedError()
