# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import devtools

from gt4py.next import common, config
from gt4py.next.iterator import builtins
from gt4py.next.iterator.builtins import BackendNotSelectedError, builtin_dispatch
from gt4py.next.program_processors import program_formatter


if TYPE_CHECKING:
    # TODO(tehrengruber): remove cirular dependency and import unconditionally
    from gt4py.next import backend as next_backend

__all__ = ["fendef", "fundef", "if_stmt", "offset", "set_at"]


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


@dataclasses.dataclass
class FendefDispatcher:
    definition: types.FunctionType
    offset_provider: Optional[common.OffsetProvider]
    column_axis: Optional[common.Dimension]

    def itir(self, *args):
        # TODO(ricoh): refactor so that `tracing` does not import this module
        #   and can be imported top level. Then set `fendef_tracing` as a
        #   proper default value, instead of using `None` as a sentinel.
        from gt4py.next.iterator.tracing import trace_fencil_definition

        fencil_definition = trace_fencil_definition(self.definition, args)

        if config.DEBUG:
            devtools.debug(fencil_definition)
        return fencil_definition

    def __call__(
        self,
        *args,
        backend: Optional[next_backend.Backend | program_formatter.ProgramFormatter] = None,
        offset_provider=None,
        column_axis=None,
    ):
        offset_provider = offset_provider or self.offset_provider
        column_axis = column_axis or self.column_axis

        if backend is not None:
            itir_node = self.itir(*args)

            # TODO(tehrengruber): remove cirular dependency and place import at the top of the file
            from gt4py.next import backend as next_backend

            if isinstance(backend, next_backend.Backend):
                assert isinstance(backend, next_backend.Backend)
                compiled_program = backend.jit(
                    itir_node, *args, offset_provider=offset_provider, column_axis=column_axis
                )
                compiled_program(*args, offset_provider=offset_provider)
            elif isinstance(backend, program_formatter.ProgramFormatter):
                return backend(
                    itir_node, *args, offset_provider=offset_provider, column_axis=column_axis
                )
            else:
                raise ValueError(
                    "Backend must be a 'gt4py.next.backend.Backend' or "
                    "'gt4py.next.program_formatter.ProgramFormatter'."
                )
        else:
            if fendef_embedded is None:
                raise RuntimeError("Embedded execution is not registered.")
            fendef_embedded(
                self.definition, *args, offset_provider=offset_provider, column_axis=column_axis
            )


def fendef(
    definition: Optional[types.FunctionType] = None,
    *,
    offset_provider: Optional[common.OffsetProvider] = None,
    column_axis: Optional[common.Dimension] = None,
):
    """
    Dispatches to embedded execution or execution with code generation.

    If `backend` keyword argument is not set or None `fendef_embedded` will be called,
    else `fendef_codegen` will be called.
    """
    if not definition:
        return functools.partial(fendef, offset_provider=offset_provider, column_axis=column_axis)

    return FendefDispatcher(
        definition=definition, offset_provider=offset_provider or {}, column_axis=column_axis
    )


def _deduce_domain(
    domain: dict[common.Dimension, range], offset_provider_type: common.OffsetProviderType
):
    if isinstance(domain, UnstructuredDomain):
        domain_builtin = builtins.unstructured_domain
    elif isinstance(domain, CartesianDomain):
        domain_builtin = builtins.cartesian_domain
    else:
        domain_builtin = (
            builtins.unstructured_domain
            if any(isinstance(o, common.ConnectivityType) for o in offset_provider_type.values())
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
                dom = _deduce_domain(dom, common.offset_provider_to_type(offset_provider))
            set_at(builtins.as_fieldop(self.fundef_dispatcher, dom)(*inps), dom, out)

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
def set_at(*args):
    return BackendNotSelectedError()


@builtin_dispatch
def if_stmt(*args):
    return BackendNotSelectedError()
