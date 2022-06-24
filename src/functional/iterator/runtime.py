import types
from dataclasses import dataclass
from typing import Callable, Optional, Union

from devtools import debug

from functional import common
from functional.fencil_processors.processor_interface import ensure_executor, ensure_formatter
from functional.iterator import builtins
from functional.iterator.builtins import BackendNotSelectedError, builtin_dispatch


__all__ = ["offset", "fundef", "fendef", "closure", "CartesianAxis"]


@dataclass(frozen=True)
class Offset:
    value: Optional[Union[int, str]] = None


def offset(value):
    return Offset(value)


# todo: rename to dimension and remove axis terminology
CartesianAxis = common.Dimension


# dependency inversion, register fendef for embedded execution or for tracing/parsing here
fendef_embedded: Optional[Callable] = None
fendef_codegen: Optional[Callable] = None


class FendefDispatcher:
    def __init__(self, function: types.FunctionType, executor_kwargs: dict):
        self.function = function
        self.out_as_kwarg_pos = executor_kwargs.pop("out_as_kwarg_pos", None)
        self.executor_kwargs = executor_kwargs

    def itir(self, *args, **kwargs):
        kwargs = self.executor_kwargs | kwargs
        fencil_definition = fendef_codegen(self.function, *args, **kwargs)
        if "debug" in kwargs:
            debug(fencil_definition)
        return fencil_definition

    def __call__(self, *args, backend=None, **kwargs):
        args, kwargs = self._rewrite_args(args, kwargs)

        if backend is not None:
            ensure_executor(backend)
            if fendef_codegen is None:
                raise RuntimeError("Backend execution is not registered")
            backend(self.itir(*args, **kwargs), *args, **kwargs)
        else:
            if fendef_embedded is None:
                raise RuntimeError("Embedded execution is not registered")
            fendef_embedded(self.function, *args, **kwargs)

    def string_format(self, *args, formatter=None, **kwargs) -> str:
        ensure_formatter(formatter)
        args, kwargs = self._rewrite_args(args, kwargs)
        return formatter(self.itir(*args, **kwargs), *args, **kwargs)

    def _rewrite_args(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if self.out_as_kwarg_pos is not None and "out" in kwargs:
            args_list = list(args)
            args_list.insert(self.out_as_kwarg_pos, kwargs.pop("out"))
            args = tuple(args_list)

        kwargs = self.executor_kwargs | kwargs

        return args, kwargs


def fendef(*dec_args, **dec_kwargs):
    """
    Dispatches to embedded execution or execution with code generation.

    If `backend` keyword argument is not set or None `fendef_embedded` will be called,
    else `fendef_codegen` will be called.
    """

    def wrapper(fun):
        return FendefDispatcher(function=fun, executor_kwargs=dec_kwargs)

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

    def __getitem__(self, domain):
        @fendef(out_as_kwarg_pos=0)
        def impl(out, *inps):
            dom = domain
            if isinstance(dom, Callable):
                # if domain is expressed as calls to builtin `domain()` we need to pass it lazily
                # as dispatching needs to happen inside of the fencil
                dom = dom()
            if isinstance(dom, dict):
                # if passed as a dict, we need to convert back to builtins for interpretation by the backends
                dom = builtins.domain(
                    *tuple(
                        map(
                            lambda x: builtins.named_range(x[0], x[1].start, x[1].stop),
                            dom.items(),
                        )
                    )
                )
            closure(dom, self, out, [*inps])

        return impl

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
