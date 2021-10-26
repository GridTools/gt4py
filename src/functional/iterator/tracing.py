import inspect
from typing import List

from functional import iterator
from eve import Node
from functional.iterator.backend_executor import execute_program
from functional.iterator.ir import (
    AxisLiteral,
    BoolLiteral,
    Expr,
    FencilDefinition,
    FloatLiteral,
    FunCall,
    FunctionDefinition,
    IntLiteral,
    Lambda,
    NoneLiteral,
    OffsetLiteral,
    Program,
    StencilClosure,
    Sym,
    SymRef,
)
from functional.iterator.runtime import CartesianAxis


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
        return FunCall(fun=SymRef(id="mul"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __rmul__(self, other):
        return make_node(other) * self

    @monkeypatch_method(Expr)
    def __truediv__(self, other):
        return FunCall(fun=SymRef(id="div"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __sub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __gt__(self, other):
        return FunCall(fun=SymRef(id="greater"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __lt__(self, other):
        return FunCall(fun=SymRef(id="less"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __call__(self, *args):
        return FunCall(fun=self, args=[*make_node(args)])


_patch_Expr()


class PatchedFunctionDefinition(FunctionDefinition):
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[*make_node(args)])


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


# builtins
@iterator.builtins.deref.register(TRACING)
def deref(arg):
    return _f("deref", arg)


@iterator.builtins.lift.register(TRACING)
def lift(sten):
    return _f("lift", sten)


@iterator.builtins.reduce.register(TRACING)
def reduce(*args):
    return _f("reduce", *args)


@iterator.builtins.scan.register(TRACING)
def scan(*args):
    return _f("scan", *args)


@iterator.builtins.is_none.register(TRACING)
def is_none(*args):
    return _f("is_none", *args)


@iterator.builtins.make_tuple.register(TRACING)
def make_tuple(*args):
    return _f("make_tuple", *args)


@iterator.builtins.nth.register(TRACING)
def nth(*args):
    return _f("nth", *args)


@iterator.builtins.compose.register(TRACING)
def compose(*args):
    return _f("compose", *args)


@iterator.builtins.domain.register(TRACING)
def domain(*args):
    return _f("domain", *args)


@iterator.builtins.named_range.register(TRACING)
def named_range(*args):
    return _f("named_range", *args)


@iterator.builtins.if_.register(TRACING)
def if_(*args):
    return _f("if_", *args)


@iterator.builtins.or_.register(TRACING)
def or_(*args):
    return _f("or_", *args)


# shift promotes its arguments to literals, therefore special
@iterator.builtins.shift.register(TRACING)
def shift(*offsets):
    offsets = tuple(OffsetLiteral(value=o) if isinstance(o, (str, int)) else o for o in offsets)
    return _f("shift", *offsets)


@iterator.builtins.plus.register(TRACING)
def plus(*args):
    return _f("plus", *args)


@iterator.builtins.minus.register(TRACING)
def minus(*args):
    return _f("minus", *args)


@iterator.builtins.mul.register(TRACING)
def mul(*args):
    return _f("mul", *args)


@iterator.builtins.div.register(TRACING)
def div(*args):
    return _f("div", *args)


@iterator.builtins.eq.register(TRACING)
def eq(*args):
    return _f("eq", *args)


@iterator.builtins.greater.register(TRACING)
def greater(*args):
    return _f("greater", *args)


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
        return BoolLiteral(value=o)
    if isinstance(o, int):
        return IntLiteral(value=o)
    if isinstance(o, float):
        return FloatLiteral(value=o)
    if isinstance(o, CartesianAxis):
        return AxisLiteral(value=o.value)
    if isinstance(o, tuple):
        return tuple(make_node(arg) for arg in o)
    if isinstance(o, list):
        return list(make_node(arg) for arg in o)
    if o is None:
        return NoneLiteral()
    if isinstance(o, iterator.runtime.FundefDispatcher):
        return SymRef(id=o.fun.__name__)
    raise NotImplementedError(f"Cannot handle {o}")


def trace_function_call(fun):
    body = fun(*list(_s(param) for param in inspect.signature(fun).parameters.keys()))
    return make_node(body) if body is not None else None


def lambdadef(fun):
    return Lambda(
        params=list(Sym(id=param) for param in inspect.signature(fun).parameters.keys()),
        expr=trace_function_call(fun),
    )


def make_function_definition(fun):
    res = PatchedFunctionDefinition(
        id=fun.__name__,
        params=list(Sym(id=param) for param in inspect.signature(fun).parameters.keys()),
        expr=trace_function_call(fun),
    )
    Tracer.add_fundef(res)
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


class Tracer:
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
def closure(domain, stencil, outputs, inputs):
    stencil(*list(_s(param) for param in inspect.signature(stencil).parameters.keys()))
    Tracer.add_closure(
        StencilClosure(
            domain=domain,
            stencil=make_node(stencil),
            outputs=outputs,
            inputs=inputs,
        )
    )


def fendef_tracing(fun, *args, **kwargs):
    with Tracer() as _:
        trace_function_call(fun)

        fencil = FencilDefinition(
            id=fun.__name__,
            params=list(Sym(id=param) for param in inspect.signature(fun).parameters.keys()),
            closures=Tracer.closures,
        )
        prog = Program(function_definitions=Tracer.fundefs, fencil_definitions=[fencil], setqs=[])
    # after tracing is done
    execute_program(prog, *args, **kwargs)


iterator.runtime.fendef_codegen = fendef_tracing
