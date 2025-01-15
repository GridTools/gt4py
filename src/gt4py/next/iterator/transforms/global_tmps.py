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
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicRange, SymbolicDomain
from gt4py.next.iterator.transforms import cse, infer_domain, inline_lambdas, collapse_tuple
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def _transform_if(
    stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> Optional[list[itir.Stmt]]:
    if isinstance(stmt, itir.SetAt) and cpm.is_call_to(stmt.expr, "if_"):
        cond, true_val, false_val = stmt.expr.args
        return [
            itir.IfStmt(
                cond=cond,
                true_branch=_transform_stmt(
                    itir.SetAt(target=stmt.target, expr=true_val, domain=stmt.domain),
                    declarations,
                    uids,
                ),
                false_branch=_transform_stmt(
                    itir.SetAt(target=stmt.target, expr=false_val, domain=stmt.domain),
                    declarations,
                    uids,
                ),
            )
        ]
    return None


# TODO(tehrengruber): This happens in _tridiagonal
def _fix_domain_starts(domain: SymbolicDomain):
    new_ranges = {}
    for dim, range_ in domain.ranges.items():
        new_ranges[dim] = SymbolicRange(
            start=im.call("maximum")(0, range_.start),
            stop=range_.stop
        )
    return SymbolicDomain(grid_type=domain.grid_type, ranges=new_ranges)


def _transform_by_pattern(
    stmt: itir.Stmt,
    predicate: Callable[[itir.Expr, int], bool],
    declarations: list[itir.Temporary],
    uids: eve_utils.UIDGenerator,
) -> Optional[list[itir.Stmt]]:
    if not isinstance(stmt, itir.SetAt):
        return None

    new_expr, extracted_fields, _ = cse.extract_subexpression(
        stmt.expr,
        predicate=predicate,
        uid_generator=eve_utils.UIDGenerator(prefix="__tmp_subexpr"),
        # TODO(tehrengruber): extracting the deepest expression first would allow us to fuse
        #  the extracted expressions resulting in fewer kernel calls & better data-locality.
        #  Extracting multiple expressions deepest-first is however not supported right now.
        # deepest_expr_first=True  # noqa: ERA001
    )

    if extracted_fields:
        tmp_stmts: list[itir.Stmt] = []

        # for each extracted expression generate:
        #  - one or more `Temporary` declarations (depending on whether the expression is a field
        #    or a tuple thereof)
        #  - one `SetAt` statement that materializes the expression into the temporary
        for tmp_sym, tmp_expr in extracted_fields.items():
            domain: infer_domain.DomainAccess = tmp_expr.annex.domain

            # TODO(tehrengruber): Implement. This happens when the expression is a combination
            #  of an `if_` call with a tuple, e.g., `if_(cond, {a, b}, {c, d})`. As long as we are
            #  able to eliminate all tuples, e.g., by propagating the scalar ifs to the top-level
            #  of a SetAt, the CollapseTuple pass will eliminate most of this cases.
            if isinstance(domain, tuple):
                # note: this change is required for the scan in _tridiagonal
                flattened_domains: tuple[domain_utils.SymbolicDomain] = tuple(
                    domain for domain in next_utils.flatten_nested_tuple(domain) if not domain is infer_domain.DomainAccessDescriptor.NEVER  # type: ignore[assignment]  # mypy not smart enough
                )
                assert len(flattened_domains) > 0
                if not all(d == flattened_domains[0] for d in flattened_domains):
                    raise NotImplementedError(
                        "Tuple expressions with different domains is not supported yet."
                    )
                domain = flattened_domains[0]
            assert isinstance(domain, domain_utils.SymbolicDomain)
            domain = _fix_domain_starts(domain)
            domain_expr = domain.as_expr()

            assert isinstance(tmp_expr.type, ts.TypeSpec)
            tmp_names: str | tuple[str | tuple, ...] = type_info.apply_to_primitive_constituents(
                lambda x: uids.sequential_id(),
                tmp_expr.type,
                tuple_constructor=lambda *elements: tuple(elements),
            )
            tmp_dtypes: (
                ts.ScalarType | ts.ListType | tuple[ts.ScalarType | ts.ListType | tuple, ...]
            ) = type_info.apply_to_primitive_constituents(
                type_info.extract_dtype,
                tmp_expr.type,
                tuple_constructor=lambda *elements: tuple(elements),
            )

            # allocate temporary for all tuple elements
            def allocate_temporary(tmp_name: str, dtype: ts.ScalarType):
                declarations.append(itir.Temporary(id=tmp_name, domain=domain_expr, dtype=dtype))  # noqa: B023 # function only used inside loop

            next_utils.tree_map(allocate_temporary)(tmp_names, tmp_dtypes)

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

            # TODO(tehrengruber): _transform_stmt not needed if deepest_expr_first=True
            tmp_stmts.extend(
                _transform_stmt(
                    itir.SetAt(target=target_expr, domain=domain_expr, expr=tmp_expr),
                    declarations,
                    uids,
                )
            )

        return [*tmp_stmts, itir.SetAt(target=stmt.target, domain=stmt.domain, expr=new_expr)]
    return None

def _is_trivial_tuple_expr(node: ir.Expr) -> bool:
    if cpm.is_call_to(node, "make_tuple"):
        return all(_is_trivial_tuple_expr(arg) for arg in node.args)
    if cpm.is_call_to(node, "tuple_get"):
        return _is_trivial_tuple_expr(node.args[1])
    if isinstance(node, (itir.SymRef, itir.Literal)):
        return True
    if cpm.is_let(node):  # TODO: write test case. occurs in test_preconditioner.py::test_scan_stencils[_tridiagonal_init_v1-_tridiagonal_init_tmp_out-test_case1]
        return _is_trivial_tuple_expr(node.fun.expr) and all(_is_trivial_tuple_expr(arg) for arg in node.args)
    return False

# occurs in _tridiagonal
def _transform_trivial_tuple_expr(stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator):
    if not isinstance(stmt, itir.SetAt):
        return None
    if cpm.is_let(stmt.expr) and any(trivial_args := [_is_trivial_tuple_expr(arg) for arg in stmt.expr.args]):
        simplified_args = [collapse_tuple.CollapseTuple.apply(
            type_inference.reinfer(arg),
            within_stencil=False,
            allow_undeclared_symbols=True,
            flags=~collapse_tuple.CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES
        ) if is_trivial else arg for arg, is_trivial in zip(stmt.expr.args, trivial_args, strict=True)]
        new_expr = im.let(*zip(stmt.expr.fun.params, simplified_args))(stmt.expr.fun.expr)
        return [itir.SetAt(
            expr=inline_lambdas.inline_lambda(new_expr, eligible_params=trivial_args),
            domain=stmt.domain,
            target=stmt.target
        )]
    return None

def _transform_stmt(
    stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> list[itir.Stmt]:
    unprocessed_stmts: list[itir.Stmt] = [stmt]
    stmts: list[itir.Stmt] = []

    transforms: list[Callable] = [
        # transform trivial tuple expr
        _transform_trivial_tuple_expr,
        # transform `if_` call into `IfStmt`
        _transform_if,
        # extract applied `as_fieldop` to top-level
        functools.partial(
            _transform_by_pattern, predicate=lambda expr, _: cpm.is_applied_as_fieldop(expr)
        ),
        # extract if_ call to the top-level
        functools.partial(
            _transform_by_pattern, predicate=lambda expr, _: cpm.is_call_to(expr, "if_")
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
    program: itir.Program,
    offset_provider: common.OffsetProvider,
    *,
    uids: Optional[eve_utils.UIDGenerator] = None,
) -> itir.Program:
    """
    Given an `itir.Program` create temporaries for intermediate values.

    This pass looks at all `as_fieldop` calls and transforms field-typed subexpressions of its
    arguments into temporaries.
    """
    program = infer_domain.infer_program(program, offset_provider=offset_provider)
    program = type_inference.infer(
        program, offset_provider_type=common.offset_provider_to_type(offset_provider)
    )

    if not uids:
        uids = eve_utils.UIDGenerator(prefix="__tmp")
    declarations = program.declarations.copy()
    new_body = []

    for stmt in program.body:
        assert isinstance(stmt, itir.SetAt)
        new_body.extend(_transform_stmt(stmt, uids=uids, declarations=declarations))

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=declarations,
        body=new_body,
    )
