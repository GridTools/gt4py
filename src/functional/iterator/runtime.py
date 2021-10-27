from dataclasses import dataclass
from typing import Optional, Union

from functional.iterator.builtins import BackendNotSelectedError, builtin_dispatch


__all__ = ["offset", "fundef", "fendef", "closure", "CartesianAxis"]


@dataclass
class Offset:
    value: Optional[Union[int, str]] = None

    def __hash__(self) -> int:
        return hash(self.value)


def offset(value):
    return Offset(value)


@dataclass
class CartesianAxis:
    value: str

    def __hash__(self) -> int:
        return hash(self.value)


# dependency inversion, register fendef for embedded execution or for tracing/parsing here
fendef_embedded = None
fendef_codegen = None


def fendef(*dec_args, **dec_kwargs):
    """
    Dispatches to embedded execution or execution with code generation.

    If `backend` keyword argument is not set or None `fendef_embedded` will be called,
    else `fendef_codegen` will be called.
    """

    def wrapper(fun):
        def impl(*args, **kwargs):
            kwargs = {**kwargs, **dec_kwargs}

            if "backend" in kwargs and kwargs["backend"] is not None:
                if fendef_codegen is None:
                    raise RuntimeError("Backend execution is not registered")
                fendef_codegen(fun, *args, **kwargs)
            else:
                if fendef_embedded is None:
                    raise RuntimeError("Embedded execution is not registered")
                fendef_embedded(fun, *args, **kwargs)

        return impl

    if len(dec_args) == 1 and len(dec_kwargs) == 0 and callable(dec_args[0]):
        return wrapper(dec_args[0])
    else:
        assert len(dec_args) == 0
        return wrapper


class FundefDispatcher:
    _hook = None
    # hook is an object that
    # - evaluates to true if it should be used,
    # - is callable with an instance of FundefDispatcher
    # - returns callable that takes the function arguments

    def __init__(self, fun) -> None:
        self.fun = fun
        self.__name__ = fun.__name__

    def __call__(self, *args):
        if type(self)._hook:
            return type(self)._hook(self)(*args)
        else:
            return self.fun(*args)

    @classmethod
    def register_hook(cls, hook):
        cls._hook = hook


def fundef(fun):
    return FundefDispatcher(fun)


@builtin_dispatch
def closure(*args):
    return BackendNotSelectedError()
