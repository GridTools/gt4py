# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import devtools

from gt4py.next import common
from gt4py.next.iterator import builtins
from gt4py.next.iterator.builtins import BackendNotSelectedError, builtin_dispatch
from gt4py.next.iterator.ir import FencilDefinition
from gt4py.next.program_processors.processor_interface import (
    ProgramExecutor,
    ProgramFormatter,
    ensure_processor_kind,
)


__all__ = ["offset", "fundef", "fendef", "closure", "set_at"]


@dataclass(frozen=True)
class Offset:
    value: Union[int, str]


def offset(value):
    return Offset(value)


class CartesianDomain(dict): ...


class UnstructuredDomain(dict): ...


# dependency inversion, register fendef for embedded execution or for tracing/parsing here
# TODO(ricoh): this pattern lead to import cycles with `fendef_codegen`
#   and was changed there. Maybe applies to `fendef_embedded` too?
fendef_embedded: Optional[Callable[[types.FunctionType], None]] = None


class FendefDispatcher:
    def __init__(self, function: types.FunctionType, executor_kwargs: dict):
        self.function = function
        self.executor_kwargs = executor_kwargs

    def itir(
        self,
        *args,
        fendef_codegen: Optional[Callable[[types.FunctionType], FencilDefinition]] = None,
        debug=False,
        **kwargs,
    ):
        args, kwargs = self._rewrite_args(args, kwargs)

        # For consistency with `__call__` we also allow these keyword arguments, but ignore them
        # here, as they are not used for code generation.
        for ignored_kwarg in ["offset_provider", "lift_mode", "column_axis"]:
            kwargs.pop(ignored_kwarg, None)

        if fendef_codegen is None:
            # TODO(ricoh): refactor so that `tracing` does not import this module
            #   and can be imported top level. Then set `fendef_tracing` as a
            #   proper default value, instead of using `None` as a sentinel.
            from gt4py.next.iterator.tracing import trace_fencil_definition

            fendef_codegen = trace_fencil_definition
        fencil_definition = fendef_codegen(self.function, args, **kwargs)
        if debug:
            devtools.debug(fencil_definition)
        return fencil_definition

    def __call__(self, *args, backend: Optional[ProgramExecutor] = None, **kwargs):
        args, kwargs = self._rewrite_args(args, kwargs)

        if backend is not None:
            ensure_processor_kind(backend, ProgramExecutor)
            backend(self.itir(*args, **kwargs), *args, **kwargs)
        else:
            if fendef_embedded is None:
                raise RuntimeError("Embedded execution is not registered.")
            fendef_embedded(self.function, *args, **kwargs)

    def format_itir(self, *args, formatter: ProgramFormatter, **kwargs) -> str:
        ensure_processor_kind(formatter, ProgramFormatter)
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
        *tuple(map(lambda x: builtins.named_range(x[0], x[1].start, x[1].stop), domain.items()))
    )


@dataclass
class FundefFencilWrapper:
    fundef_dispatcher: FundefDispatcher
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
def closure(*args):  # TODO remove
    return BackendNotSelectedError()


@builtin_dispatch
def set_at(*args):
    return BackendNotSelectedError()
