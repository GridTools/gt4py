# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Callable, Optional

from gt4py.eve import utils as eve_utils
from gt4py.next import common, utils as next_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import cse, infer_domain, inline_lambdas
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def transform_if(
    stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> Optional[list[itir.Stmt]]:
    if not isinstance(stmt, itir.SetAt):
        return None

    if cpm.is_call_to(stmt.expr, "if_"):
        cond, true_val, false_val = stmt.expr.args
        return [
            itir.IfStmt(
                cond=cond,
                # recursively transform
                true_branch=transform(
                    itir.SetAt(target=stmt.target, expr=true_val, domain=stmt.domain),
                    declarations,
                    uids,
                ),
                false_branch=transform(
                    itir.SetAt(target=stmt.target, expr=false_val, domain=stmt.domain),
                    declarations,
                    uids,
                ),
            )
        ]
    return None


def transform_by_pattern(
    stmt: itir.Stmt, predicate, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> Optional[list[itir.Stmt]]:
    if not isinstance(stmt, itir.SetAt):
        return None

    new_expr, extracted_fields, _ = cse.extract_subexpression(
        stmt.expr,
        predicate=predicate,
        uid_generator=eve_utils.UIDGenerator(prefix="__tmp_subexpr"),
        # TODO(tehrengruber): extracting the deepest expression first would allow us to fuse
        #  the extracted expressions resulting in fewer kernel calls, better data-locality.
        #  Extracting the multiple expressions deepest-first is however not supported right now.
        # deepest_expr_first=True  # noqa: ERA001
    )

    if extracted_fields:
        tmp_stmts: list[itir.Stmt] = []

        # for each extracted expression generate:
        #  - one or more `Temporary` declarations (depending on whether the expression is a field
        #    or a tuple thereof)
        #  - one `SetAt` statement that materializes the expression into the temporary
        for tmp_sym, tmp_expr in extracted_fields.items():
            domain = tmp_expr.annex.domain

            # TODO(tehrengruber): Implement. This happens when the expression for a combination
            #  of an `if_` call with a tuple, e.g., `if_(cond, {a, b}, {c, d})`. As long as we are
            #  able to eliminate all tuples, e.g., by propagating the scalar ifs to the top-level
            #  of a SetAt, the CollapseTuple pass will eliminate most of this cases.
            if isinstance(domain, tuple):
                flattened_domains: tuple[itir.Expr] = next_utils.flatten_nested_tuple(domain)  # type: ignore[assignment]  # mypy not smart enough
                if not all(d == flattened_domains[0] for d in flattened_domains):
                    raise NotImplementedError(
                        "Tuple expressions with different domains is not " "supported yet."
                    )
                domain = flattened_domains[0]

            assert isinstance(tmp_expr.type, ts.TypeSpec)
            tmp_names: str | tuple[str | tuple, ...] = type_info.apply_to_primitive_constituents(
                lambda x: uids.sequential_id(),
                tmp_expr.type,
                tuple_constructor=lambda *elements: tuple(elements),
            )
            tmp_dtypes: ts.ScalarType | tuple[ts.ScalarType | tuple, ...] = (
                type_info.apply_to_primitive_constituents(
                    type_info.extract_dtype,
                    tmp_expr.type,
                    tuple_constructor=lambda *elements: tuple(elements),
                )
            )

            # allocate temporary for all tuple elements
            def allocate_temporary(tmp_name: str, dtype: ts.ScalarType, domain: itir.Expr):
                declarations.append(itir.Temporary(id=tmp_name, domain=domain, dtype=dtype))

            next_utils.tree_map(functools.partial(allocate_temporary, domain=domain))(
                tmp_names, tmp_dtypes
            )

            # if the expr is a field this just gives a simple `itir.SymRef`, otherwise we generate a
            #  `make_tuple` expression.
            target_expr: itir.Expr = next_utils.tree_map(
                lambda x: im.ref(x), result_collection_constructor=lambda els: im.make_tuple(*els)
            )(tmp_names)  # type: ignore[assignment]  # typing of tree_map does not reflect action of `result_collection_constructor` yet

            # note: the let would be removed automatically by the `cse.extract_subexpression`, but
            # we remove it here for readability & debuggability.
            new_expr = inline_lambdas.inline_lambda(
                im.let(tmp_sym, target_expr)(new_expr), opcount_preserving=False
            )

            # TODO: transform not needed if deepest_expr_first=True
            tmp_stmts.extend(
                transform(
                    itir.SetAt(target=target_expr, domain=domain, expr=tmp_expr), declarations, uids
                )
            )

        return [*tmp_stmts, itir.SetAt(target=stmt.target, domain=stmt.domain, expr=new_expr)]
    return None


def transform(
    stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> list[itir.Stmt]:
    unprocessed_stmts: list[itir.Stmt] = [stmt]
    stmts: list[itir.Stmt] = []

    transforms: list[Callable] = [
        # transform functional if_ into if-stmt
        transform_if,
        # extract applied `as_fieldop` to top-level
        functools.partial(
            transform_by_pattern, predicate=lambda expr, _: cpm.is_applied_as_fieldop(expr)
        ),
        # extract functional if_ to the top-level
        functools.partial(
            transform_by_pattern, predicate=lambda expr, _: cpm.is_call_to(expr, "if_")
        ),
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


def create_global_tmps(
    program: itir.Program, offset_provider: common.OffsetProvider
) -> itir.Program:
    program = infer_domain.infer_program(program, offset_provider)
    program = type_inference.infer(program, offset_provider=offset_provider)

    uids = eve_utils.UIDGenerator(prefix="__tmp")
    declarations = program.declarations.copy()
    new_body = []

    for stmt in program.body:
        assert isinstance(stmt, itir.SetAt)
        new_body.extend(transform(stmt, uids=uids, declarations=declarations))

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=declarations,
        body=new_body,
    )
