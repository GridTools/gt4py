from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

from iterator.builtins import BackendNotSelectedError, builtin_dispatch


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


fendef_registry: Dict[Optional[Callable], Callable] = {}


# TODO the dispatching is linear, not sure if there is an easy way to make it constant
def fendef(*dec_args, **dec_kwargs):
    def wrapper(fun):
        def impl(*args, **kwargs):
            kwargs = {**kwargs, **dec_kwargs}

            for key, val in fendef_registry.items():
                if key is not None and key(kwargs):
                    val(fun, *args, **kwargs)
                    return
            if None in fendef_registry:
                fendef_registry[None](fun, *args, **kwargs)
                return
            raise RuntimeError("Unreachable")

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
