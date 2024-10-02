# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Mapping
from typing import Any, Callable, Final, Iterable, Literal, Optional, Sequence

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.type_system import type_info
from gt4py.next.iterator.transforms import inline_lambdas

# TODO: remove
SimpleTemporaryExtractionHeuristics = None
CreateGlobalTmps = None

from gt4py.next.iterator.transforms import cse

class IncompleteTemporary:
    expr: itir.Expr
    target: itir.Expr

def get_expr_domain(expr: itir.Expr, ctx=None):
    ctx = ctx or {}

    if cpm.is_applied_as_fieldop(expr):
        _, domain = expr.fun.args
        return domain
    elif cpm.is_call_to(expr, "tuple_get"):
        idx_expr, tuple_expr = expr.args
        assert isinstance(idx_expr, itir.Literal) and type_info.is_integer(idx_expr.type)
        idx = int(idx_expr.value)
        tuple_expr_domain = get_expr_domain(tuple_expr, ctx)
        assert isinstance(tuple_expr_domain, tuple) and idx < len(tuple_expr_domain)
        return tuple_expr_domain[idx]
    elif cpm.is_call_to(expr, "make_tuple"):
        return tuple(get_expr_domain(el, ctx) for el in expr.args)
    elif cpm.is_call_to(expr, "if_"):
        cond, true_val, false_val = expr.args
        true_domain, false_domain = get_expr_domain(true_val, ctx), get_expr_domain(false_val, ctx)
        assert true_domain == false_domain
        return true_domain
    elif cpm.is_let(expr):
        new_ctx = {}
        for var_name, var_value in zip(expr.fun.params, expr.args, strict=True):
            new_ctx[var_name.id] = get_expr_domain(var_value, ctx)
        return get_expr_domain(expr.fun.expr, ctx={**ctx, **new_ctx})
    raise ValueError()


def transform_if(stmt: itir.SetAt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator):
    if not isinstance(stmt, itir.SetAt):
        return None

    if cpm.is_call_to(stmt.expr, "if_"):
        cond, true_val, false_val = stmt.expr.args
        return [itir.IfStmt(
            cond=cond,
            # recursively transform
            true_branch=transform(itir.SetAt(target=stmt.target, expr=true_val, domain=stmt.domain), declarations, uids),
            false_branch=transform(itir.SetAt(target=stmt.target, expr=false_val, domain=stmt.domain), declarations, uids),
        )]
    return None

def transform_by_pattern(stmt: itir.SetAt, predicate, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator):
    if not isinstance(stmt, itir.SetAt):
        return None

    new_expr, extracted_fields, _ = cse.extract_subexpression(
        stmt.expr,
        predicate=predicate,
        uid_generator=uids,
        # allows better fusing later on
        #deepest_expr_first=True  # TODO: better, but not supported right now
    )

    if extracted_fields:
        new_stmts = []
        for tmp_sym, tmp_expr in extracted_fields.items():
            # TODO: expr domain can not be a tuple here
            domain = get_expr_domain(tmp_expr)

            scalar_type = type_info.apply_to_primitive_constituents(
                type_info.extract_dtype, tmp_expr.type
            )
            declarations.append(itir.Temporary(id=tmp_sym.id, domain=domain, dtype=scalar_type))

            # TODO: transform not needed if deepest_expr_first=True
            new_stmts.extend(transform(itir.SetAt(target=im.ref(tmp_sym.id), domain=domain, expr=tmp_expr), declarations, uids))

        return [
            *new_stmts,
            itir.SetAt(
                target=stmt.target,
                domain=stmt.domain,
                expr=new_expr
            )
        ]
    return None

def transform(stmt: itir.SetAt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator):
    # TODO: what happens for a trivial let, e.g `let a=as_fieldop() in a end`?
    unprocessed_stmts = [stmt]
    stmts = []

    transforms = [
        # transform functional if_ into if-stmt
        transform_if,
        # extract applied `as_fieldop` to top-level
        functools.partial(transform_by_pattern, predicate=lambda expr, _: cpm.is_applied_as_fieldop(expr)),
        # extract functional if_ to the top-level
        functools.partial(transform_by_pattern, predicate=lambda expr, _: cpm.is_call_to(expr, "if_")),
    ]

    while unprocessed_stmts:
        stmt = unprocessed_stmts.pop(0)

        did_transform = False
        for transform in transforms:
            transformed_stmts = transform(stmt=stmt, declarations=declarations, uids=uids)
            if transformed_stmts:
                unprocessed_stmts = [*transformed_stmts, *unprocessed_stmts]
                did_transform = True
                break

        # no transformation occurred
        if not did_transform:
            stmts.append(stmt)

    return stmts

def create_global_tmps(program: itir.Program):
    uids = eve_utils.UIDGenerator(prefix="__tmp")
    declarations = program.declarations
    new_body = []

    for stmt in program.body:
        if isinstance(stmt, (itir.SetAt, itir.IfStmt)):
            new_body.extend(
                transform(stmt, uids=uids, declarations=declarations)
            )
        else:
            raise NotImplementedError()

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=declarations,
        body=new_body
    )