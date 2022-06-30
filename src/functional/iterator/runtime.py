from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from devtools import debug

from functional import common
from functional.iterator import builtins
from functional.iterator.builtins import BackendNotSelectedError, builtin_dispatch
from functional.iterator.processor_interface import (
    FencilExecutor,
    FencilFormatter,
    ensure_executor,
    ensure_formatter,
)


__all__ = ["offset", "fundef", "fendef", "closure", "CartesianAxis"]


@dataclass(frozen=True)
class Offset:
    value: Union[int, str]


def offset(value):
    return Offset(value)


# todo: rename to dimension and remove axis terminology
CartesianAxis = common.Dimension


class CartesianDomain(dict):
    ...


class UnstructuredDomain(dict):
    ...


# dependency inversion, register fendef for embedded execution or for tracing/parsing here
fendef_embedded: Optional[Callable] = None
fendef_codegen: Optional[Callable] = None


class FendefDispatcher:
    def __init__(self, function: types.FunctionType, executor_kwargs: dict):
        self.function = function
        self.executor_kwargs = executor_kwargs

    def itir(self, *args, **kwargs):
        args, kwargs = self._rewrite_args(args, kwargs)
        if fendef_codegen is None:
            raise RuntimeError("Backend execution is not registered")
        fencil_definition = fendef_codegen(self.function, *args, **kwargs)
        if "debug" in kwargs:
            debug(fencil_definition)
        return fencil_definition

    def __call__(self, *args, backend: Optional[FencilExecutor] = None, **kwargs):
        args, kwargs = self._rewrite_args(args, kwargs)

        if backend is not None:
            ensure_executor(backend)
            backend(self.itir(*args, **kwargs), *args, **kwargs)
        else:
            if fendef_embedded is None:
                raise RuntimeError("Embedded execution is not registered")
            fendef_embedded(self.function, *args, **kwargs)

    def format_itir(self, *args, formatter: FencilFormatter, **kwargs) -> str:
        ensure_formatter(formatter)
        args, kwargs = self._rewrite_args(args, kwargs)
        return formatter(self.itir(*args, **kwargs), *args, **kwargs)

    def _rewrite_args(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
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


def _deduce_domain(domain: dict[common.Dimension, range], offset_provider: dict[str, Any]):
    if isinstance(domain, UnstructuredDomain):
        domain_builtin = builtins.unstructured_domain
    elif isinstance(domain, CartesianDomain):
        domain_builtin = builtins.cartesian_domain
    else:
        domain_builtin = (
            builtins.unstructured_domain
            if any(isinstance(o, common.Connectivity) for o in offset_provider.values())
            else builtins.cartesian_domain
        )

    return domain_builtin(
        *tuple(
            map(
                lambda x: builtins.named_range(x[0], x[1].start, x[1].stop),
                domain.items(),
            )
        )
    )


@dataclass
class FundefFencilWrapper:
    fundef_dispatcher: FendefDispatcher
    domain: Callable | dict

    def _get_fencil(self, offset_provider=None):
        @fendef
        def impl(out, *inps):
            dom = self.domain
            if isinstance(dom, Callable):
                # if domain is expressed as calls to builtin `domain()` we need to pass it lazily
                # as dispatching needs to happen inside of the fencil
                dom = dom()
            elif isinstance(dom, dict):
                # if passed as a dict, we need to convert back to builtins for interpretation by the backends
                assert offset_provider is not None
                dom = _deduce_domain(dom, offset_provider)
            closure(dom, self.fundef_dispatcher, out, [*inps])

        return impl

    def itir(self, *args, **kwargs):
        return self._get_fencil(offset_provider=kwargs.get("offset_provider")).itir(*args, **kwargs)

    def __call__(self, *args, out, **kwargs):
        self._get_fencil(offset_provider=kwargs.get("offset_provider"))(out, *args, **kwargs)

    def format_itir(self, *args, out, **kwargs):
        return self._get_fencil(offset_provider=kwargs.get("offset_provider")).format_itir(
            out, *args, **kwargs
        )


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
        return FundefFencilWrapper(self, domain)

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
