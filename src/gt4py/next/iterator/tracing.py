# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import inspect
import typing
from typing import List

from gt4py._core import definitions as core_defs
from gt4py.eve import Node
from gt4py.next import common, iterator
from gt4py.next.iterator import builtins
from gt4py.next.iterator.ir import (
    AxisLiteral,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    Lambda,
    NoneLiteral,
    OffsetLiteral,
    StencilClosure,
    Sym,
    SymRef,
)
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications, type_translation


TRACING = "tracing"


def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


def _patch_Expr():
    @monkeypatch_method(Expr)
    def __add__(self, other):
        return FunCall(fun=SymRef(id="plus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __radd__(self, other):
        return make_node(other) + self

    @monkeypatch_method(Expr)
    def __mul__(self, other):
        return FunCall(fun=SymRef(id="multiplies"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __rmul__(self, other):
        return make_node(other) * self

    @monkeypatch_method(Expr)
    def __truediv__(self, other):
        return FunCall(fun=SymRef(id="divides"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __sub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __rsub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[make_node(other), self])

    @monkeypatch_method(Expr)
    def __gt__(self, other):
        return FunCall(fun=SymRef(id="greater"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __lt__(self, other):
        return FunCall(fun=SymRef(id="less"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __call__(self, *args):
        return FunCall(fun=self, args=[make_node(arg) for arg in args])


def _patch_FunctionDefinition():
    @monkeypatch_method(FunctionDefinition)
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[make_node(arg) for arg in args])


_patch_Expr()
_patch_FunctionDefinition()


def _s(id_):
    return SymRef(id=id_)


def trace_function_argument(arg):
    if isinstance(arg, iterator.runtime.FundefDispatcher):
        make_function_definition(arg.fun)
        return _s(arg.fun.__name__)
    return arg


def _f(fun, *args):
    if isinstance(fun, str):
        fun = _s(fun)

    args = [trace_function_argument(arg) for arg in args]
    return FunCall(fun=fun, args=[make_node(arg) for arg in args])


# shift promotes its arguments to literals, therefore special
@iterator.builtins.shift.register(TRACING)
def shift(*offsets):
    offsets = tuple(OffsetLiteral(value=o) if isinstance(o, (str, int)) else o for o in offsets)
    return _f("shift", *offsets)


@dataclasses.dataclass(frozen=True)
class BuiltinTracer:
    name: str

    def __call__(self, *args):
        return _f(self.name, *args)


for builtin_name in builtins.BUILTINS:
    if builtin_name == "shift":
        continue

    decorator = getattr(iterator.builtins, builtin_name).register(TRACING)
    decorator(BuiltinTracer(name=builtin_name))


# helpers
def make_node(o):
    if isinstance(o, Node):
        return o
    if callable(o):
        if o.__name__ == "<lambda>":
            return lambdadef(o)
        if hasattr(o, "__code__") and o.__code__.co_flags & inspect.CO_NESTED:
            return lambdadef(o)
    if isinstance(o, iterator.runtime.Offset):
        return OffsetLiteral(value=o.value)
    if isinstance(o, core_defs.Scalar):
        return im.literal_from_value(o)
    if isinstance(o, common.Dimension):
        return AxisLiteral(value=o.value)
    if isinstance(o, tuple):
        return _f("make_tuple", *(make_node(arg) for arg in o))
    if o is None:
        return NoneLiteral()
    if hasattr(o, "fun"):
        return SymRef(id=o.fun.__name__)
    raise NotImplementedError(f"Cannot handle '{o}'.")


def trace_function_call(fun, *, args=None):
    if args is None:
        args = (_s(param) for param in inspect.signature(fun).parameters.keys())
    body = fun(*list(args))

    return make_node(body) if body is not None else None


def lambdadef(fun):
    return Lambda(
        params=list(Sym(id=param) for param in inspect.signature(fun).parameters.keys()),
        expr=trace_function_call(fun),
    )


def make_function_definition(fun):
    res = FunctionDefinition(
        id=fun.__name__,
        params=list(Sym(id=param) for param in inspect.signature(fun).parameters.keys()),
        expr=trace_function_call(fun),
    )
    TracerContext.add_fundef(res)
    return res


class FundefTracer:
    def __call__(self, fundef_dispatcher: iterator.runtime.FundefDispatcher):
        def fun(*args):
            res = make_function_definition(fundef_dispatcher.fun)
            return res(*args)

        return fun

    def __bool__(self):
        return iterator.builtins.builtin_dispatch.key == TRACING


iterator.runtime.FundefDispatcher.register_hook(FundefTracer())


class TracerContext:
    fundefs: List[FunctionDefinition] = []
    closures: List[StencilClosure] = []

    @classmethod
    def add_fundef(cls, fun):
        if fun not in cls.fundefs:
            cls.fundefs.append(fun)

    @classmethod
    def add_closure(cls, closure):
        cls.closures.append(closure)

    def __enter__(self):
        iterator.builtins.builtin_dispatch.push_key(TRACING)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        type(self).fundefs = []
        type(self).closures = []
        iterator.builtins.builtin_dispatch.pop_key()


@iterator.runtime.closure.register(TRACING)
def closure(domain, stencil, output, inputs):
    if hasattr(stencil, "__name__") and stencil.__name__ in iterator.builtins.__all__:
        stencil = _s(stencil.__name__)
    else:
        stencil(*(_s(param) for param in inspect.signature(stencil).parameters))
        stencil = make_node(stencil)
    TracerContext.add_closure(
        StencilClosure(
            domain=domain,
            stencil=stencil,
            output=output,
            inputs=inputs,
        )
    )


def _contains_tuple_dtype_field(arg):
    if isinstance(arg, tuple):
        return any(_contains_tuple_dtype_field(el) for el in arg)
    # TODO(tehrengruber): The LocatedField protocol does not have a dtype property and the
    #  various implementations have different behaviour (some return e.g. `np.dtype("int32")`
    #  other `np.int32`). We just ignore the error here and postpone fixing this to when
    #  the new storages land (The implementation here works for LocatedFieldImpl).

    return common.is_field(arg) and any(dim is None for dim in arg.domain.dims)


def _make_fencil_params(fun, args, *, use_arg_types: bool) -> list[Sym]:
    params: list[Sym] = []
    param_infos = list(inspect.signature(fun).parameters.values())

    for i, arg in enumerate(args):
        if i < len(param_infos):
            param_info = param_infos[i]
        else:
            if param_info.kind != inspect.Parameter.VAR_POSITIONAL:
                # the last parameter info might also be a keyword or variadic keyword argument, but
                # they are not supported.
                raise NotImplementedError(
                    "Only 'POSITIONAL_OR_KEYWORD' or 'VAR_POSITIONAL' parameters are supported."
                )
            param_info = param_infos[-1]

        if param_info.kind == inspect.Parameter.VAR_POSITIONAL:
            param_name = f"_{param_info.name}{i}"
        elif param_info.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            param_name = param_info.name
        else:
            raise NotImplementedError(
                "Only 'POSITIONAL_OR_KEYWORD' or 'VAR_POSITIONAL' parameters are supported."
            )

        kind, dtype = None, None
        if use_arg_types:
            # TODO(tehrengruber): Fields of tuples are not supported yet. Just ignore them for now.
            if not _contains_tuple_dtype_field(arg):
                arg_type = type_translation.from_value(arg)
                # TODO(tehrengruber): Support more types.
                if isinstance(arg_type, type_specifications.FieldType):
                    kind = "Iterator"
                    dtype = (
                        arg_type.dtype.kind.name.lower(),  # actual dtype
                        type_info.is_local_field(arg_type),  # is list
                    )

        params.append(Sym(id=param_name, kind=kind, dtype=dtype))
    return params


def trace_fencil_definition(
    fun: typing.Callable, args: typing.Iterable, *, use_arg_types=True
) -> FencilDefinition:
    """
    Transform fencil given as a callable into `itir.FencilDefinition` using tracing.

    Arguments:
        fun: The fencil / callable to trace.
        args: A list of arguments, e.g. fields, scalars or composites thereof. If `use_arg_types`
            is `False` may also be dummy values.

    Keyword arguments:
        use_arg_types: Deduce type of the arguments and add them to the fencil parameter nodes
            (i.e. `itir.Sym`s).
    """
    with TracerContext() as _:
        params = _make_fencil_params(fun, args, use_arg_types=use_arg_types)
        trace_function_call(fun, args=(_s(param.id) for param in params))

        return FencilDefinition(
            id=fun.__name__,
            function_definitions=TracerContext.fundefs,
            params=params,
            closures=TracerContext.closures,
        )
