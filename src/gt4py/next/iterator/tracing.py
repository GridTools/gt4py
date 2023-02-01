# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from typing import List

from gt4py.eve import Node
from gt4py.next import iterator
from gt4py.next.iterator import builtins
from gt4py.next.iterator.ir import (
    AxisLiteral,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    Lambda,
    Literal,
    NoneLiteral,
    OffsetLiteral,
    StencilClosure,
    Sym,
    SymRef,
)
from gt4py.next.iterator.runtime import CartesianAxis


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
        return FunCall(fun=self, args=[*make_node(args)])


def _patch_FunctionDefinition():
    @monkeypatch_method(FunctionDefinition)
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[*make_node(args)])


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
    return FunCall(fun=fun, args=[*make_node(args)])


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
    if isinstance(o, bool):
        return Literal(value=str(o), type="bool")
    if isinstance(o, int):
        return Literal(value=str(o), type="int")
    if isinstance(o, float):
        return Literal(value=str(o), type="float")
    if isinstance(o, CartesianAxis):
        return AxisLiteral(value=o.value)
    if isinstance(o, tuple):
        return tuple(make_node(arg) for arg in o)
    if isinstance(o, list):
        return list(make_node(arg) for arg in o)
    if o is None:
        return NoneLiteral()
    if hasattr(o, "fun"):
        return SymRef(id=o.fun.__name__)
    raise NotImplementedError(f"Cannot handle {o}")


def trace_function_call(fun, *, args=None):
    if args is None:
        args = (_s(param) for param in inspect.signature(fun).parameters.keys())
    body = fun(*list(args))

    if isinstance(body, tuple):
        return _f("make_tuple", *tuple(make_node(b) for b in body))
    else:
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


def _make_param_names(fun, args):
    """Expand *args parameter with remaining args."""
    args = [*args]
    param_names = []
    for p in inspect.signature(fun).parameters.values():
        if (
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or p.kind == inspect.Parameter.POSITIONAL_ONLY
        ):
            args.pop(0)
            param_names.append(p.name)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            for i in range(len(args)):
                param_names.append(f"_var{i}")
        else:
            raise RuntimeError("Illegal parameter kind")
    return param_names


def trace(fun, args):
    with TracerContext() as _:
        param_names = _make_param_names(fun, args)
        trace_function_call(fun, args=(_s(p) for p in param_names))

        return FencilDefinition(
            id=fun.__name__,
            function_definitions=TracerContext.fundefs,
            params=list(Sym(id=param) for param in param_names),
            closures=TracerContext.closures,
        )


def fendef_tracing(fun, *args, **kwargs) -> FencilDefinition:
    return trace(fun, args=args)
