# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools

import math
from typing import Callable, Iterable, TypeVar, Union, cast

import ordered_set

import gt4py.next.iterator.ir_utils.ir_makers as im
from gt4py.eve import (
    NodeTranslator,
    NodeVisitor,
    PreserveLocationVisitor,
    SymbolTableTrait,
    VisitorWithSymbolTableTrait,
    utils as eve_utils,
)
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_info, type_specifications as ts

def bindings_to_let(
    result: itir.Expr, bindings: dict[str, itir.Expr], deps: dict[str, ordered_set.OrderedSet[str]]
) -> itir.Expr:
    """Reassemble the ``bindings``/``deps`` of a `ForwardPass` into a let expression.

    Bindings are grouped by dependency level: a binding sits one level deeper
    than the deepest binding it depends on (free variables count as level -1).
    All bindings on the same level are independent of each other and are
    therefore emitted in a single ``let`` so the resulting expression stays
    shallow instead of nesting one ``let`` per binding.
    """
    binding_names = set(bindings)

    level: dict[str, int] = {}

    def compute_level(name: str) -> int:
        if name not in level:
            parents = [compute_level(d) for d in deps[name] if d in binding_names]
            level[name] = 1 + max(parents) if parents else 0
        return level[name]

    grouped: dict[int, list[str]] = {}
    for name in bindings:  # preserve creation order within a level
        grouped.setdefault(compute_level(name), []).append(name)

    expr = result
    for lvl in sorted(grouped, reverse=True):  # innermost (deepest) let first
        expr = im.let(*((name, bindings[name]) for name in grouped[lvl]))(expr)
    return expr

class ForwardPass(NodeTranslator):
    """Flatten an expression into a sequence of let-bindings (A-normal form).

    Every (nested) `FunCall` is hoisted into a freshly named binding whose
    arguments are all `SymRef`s or `Literal`s, such that no binding contains a
    nested call anymore. The visitor threads two accumulators through the
    traversal:

    - ``bindings``: maps each generated (or let-bound) name to its flat
      right-hand side, in the order the bindings are created.
    - ``deps``: maps each name to the ordered set of ``SymRef`` names its
      right-hand side references (both other bindings and free variables). This
      is the data-dependency information the backward pass consumes.

    Instead of returning a rewritten tree, ``visit`` returns a ``SymRef`` to the
    binding holding the (sub)expression's result.
    """

    @classmethod
    def apply(
        cls, node: itir.Expr
    ) -> tuple[
        itir.SymRef | itir.Literal, dict[str, itir.Expr], dict[str, ordered_set.OrderedSet[str]]
    ]:
        bindings: dict[str, itir.Expr] = {}
        deps: dict[str, ordered_set.OrderedSet[str]] = {}
        result = cls().visit(node, bindings=bindings, deps=deps)
        return result, bindings, deps

    def __init__(self) -> None:
        self._counter = itertools.count()

    def _fresh_name(self) -> str:
        return f"__ad_fwd_{next(self._counter)}"

    def visit_Node(self, node: itir.Node):
        raise NotImplementedError()

    def visit_FunCall(
        self,
        node: itir.FunCall,
        *,
        bindings: dict[str, itir.Expr],
        deps: dict[str, ordered_set.OrderedSet[str]],
    ) -> itir.SymRef:
        new_args = tuple(self.visit(arg, bindings=bindings, deps=deps) for arg in node.args)

        if cpm.is_let(node):
            for param, new_arg in zip(node.fun.params, new_args, strict=True):
                assert param.id not in bindings  # TODO: shadowing not supported
                bindings[param.id] = new_arg
                deps[param.id] = ordered_set.OrderedSet(
                    [new_arg.id] if isinstance(new_arg, itir.SymRef) else []
                )
            return self.visit(node.fun.expr, bindings=bindings, deps=deps)

        new_name = self._fresh_name()
        bindings[new_name] = im.call(node.fun)(*new_args)
        deps[new_name] = ordered_set.OrderedSet(
            arg.id for arg in new_args if isinstance(arg, itir.SymRef)
        )
        return im.ref(new_name)

    def visit_SymRef(
        self,
        node: itir.SymRef,
        *,
        bindings: dict[str, itir.Expr],
        deps: dict[str, ordered_set.OrderedSet[str]],
    ) -> itir.SymRef:
        return node

    def visit_Literal(self, node: itir.Literal, **kwargs) -> itir.Literal:
        return node


def diff(expr: itir.Expr, var_name: str):
    if isinstance(expr, itir.FunCall):
        f, args = expr.fun, expr.args
        if cpm.is_call_to(expr, "plus"):
            return im.plus(*(diff(arg, var_name) for arg in args))
        elif cpm.is_call_to(expr, "minus"):
            return im.minus(*(diff(arg, var_name) for arg in args))
        elif cpm.is_call_to(expr, "multiplies"):
            return im.plus(
                im.multiplies_(args[0], diff(args[1], var_name)),
                im.multiplies_(diff(args[0], var_name), args[1])
            )
        elif cpm.is_call_to(expr, ("sin", "cos")):
            arg, = args
            if f.id == "sin":
                g = im.call("cos")
            elif f.id == "cos":
                g = lambda x: im.call("neg")(im.call("sin")(x))
            return im.multiplies_(g(arg), diff(arg, var_name))
        elif cpm.is_call_to(expr, "make_tuple"):
            return im.make_tuple(*(diff(arg, var_name) for arg in args))
    elif isinstance(expr, itir.SymRef):
        if expr.id == var_name:
            return im.literal_from_value(1)
        return im.literal_from_value(0)
    elif isinstance(expr, itir.Literal):
        return im.literal_from_value(0)
    raise NotImplementedError()


# TODO: let f=lambda ... in ... end is not supported. Throw an error
def backward_diff(f: itir.Lambda, argnums: tuple[int] | None = None):  # TODO: impl args arg
    adjoints: dict[str, itir.Expr] = {}
    adjoint_deps: dict[str, ordered_set.OrderedSet[str]] = {}

    get_adjoint = lambda name: f"__adjoint_{name}"

    primals_out, bindings, deps = ForwardPass.apply(f.expr)

    for param in f.params:
        deps[param.id] = ordered_set.OrderedSet([])
        # initialize adjoints so that they are always defined, regardless of params being referenced
        adjoints[get_adjoint(param)] = im.literal_from_value(0)
        adjoint_deps[get_adjoint(param)] = ordered_set.OrderedSet()

    adjoints[get_adjoint(primals_out.id)] = im.literal_from_value(1)
    adjoint_deps[get_adjoint(primals_out.id)] = ordered_set.OrderedSet()
    bindings_to_process: ordered_set.OrderedSet[str] = ordered_set.OrderedSet([primals_out.id])
    while bindings_to_process:
        binding = bindings_to_process.pop()
        adjoint = get_adjoint(binding)
        bindings_to_process.update(deps[binding])
        for dep in deps[binding]:
            derivative = diff(bindings[binding], dep)
            derivative = ConstantFolding.apply(derivative)  # todo: revisit
            adjoint_contribution = im.multiplies_(adjoint, derivative)
            if get_adjoint(dep) in adjoints:
                adjoints[get_adjoint(dep)] = im.plus(adjoints[get_adjoint(dep)], adjoint_contribution)
            else:
                adjoints[get_adjoint(dep)] = adjoint_contribution
            adjoint_deps.setdefault(get_adjoint(dep), ordered_set.OrderedSet())
            adjoint_deps[get_adjoint(dep)].add(adjoint)
            # the derivative and hence also the adjoint contains references to the deps of the
            # binding (which is equivalent to adding a dependency to the binding itself)
            adjoint_deps[get_adjoint(dep)].add(binding)

    assert not (adjoints.keys() & bindings.keys())
    assert not (adjoint_deps.keys() & deps.keys())

    return primals_out, im.lambda_(*f.params)(bindings_to_let(
        im.make_tuple(*(adjoints[f"__adjoint_{param.id}"] for i, param in enumerate(f.params) if i in (argnums or range(len(f.params))))),
        {**bindings, **adjoints},
        {**deps, **adjoint_deps}
    ))

def vjp(f: itir.Lambda, argnums: tuple[int] | None = None):
    # A bare lambda only gets a `FunctionType` once it is applied, so infer the type by
    # applying it to placeholder arguments of the (already typed) parameters.
    f = itir_type_inference.infer(
        im.call(f)(*(im.literal("0", param.type) for param in f.params)),  # todo: remove type inference hack
        offset_provider_type={},
        allow_undeclared_symbols=True,
    ).fun
    assert isinstance(f.type.returns, ts.TupleType)
    dim_input, dim_output = len(f.params), len(f.type.returns.types)
    primals_out, jacobian = backward_diff(f, argnums)
    jacobian = itir_type_inference.infer(
        im.call(jacobian)(*(im.literal("0", param.type) for param in f.params)),  # todo: remove type inference hack
        offset_provider_type={},
        allow_undeclared_symbols=True,
    ).fun

    jacobian_eval = im.call(jacobian)(*[param.id for param in f.params])

    assert all(isinstance(tt, ts.TupleType) and len(tt.types) == dim_input for tt in jacobian.type.returns.elements)

    result_els = []
    for i in range(dim_input):
        result_els.append(functools.reduce(im.plus, [im.multiplies_(im.tuple_get(j, "v"), im.tuple_get(j, im.tuple_get(i, jacobian_eval))) for j in range(dim_output)]))

    return im.lambda_("v")(im.lambda_(*f.params)(im.make_tuple(*result_els)))

class BackwardDiff(NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "grad"):  # grad(lambda x: ..., {0, 1, 2, ...})
            assert len(node.args) == 2
            f: itir.Lambda = node.args[0]
            assert isinstance(f, itir.Lambda)
            assert cpm.is_call_to(node.args[1], "make_tuple") and all(isinstance(arg, itir.Literal) and type_info.is_integral_scalar(arg.type) for arg in node.args[1].args)
            return backward_diff(f)
        return node