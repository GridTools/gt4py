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
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicDomain
from gt4py.next.iterator.transforms import cse, infer_domain, inline_lambdas
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def _filter_setat_by_domain(stmt: itir.SetAt, domain: SymbolicDomain) -> itir.SetAt:
    assert cpm.is_call_to(stmt.expr, "make_tuple")

    new_targets = []
    new_els = []
    # we cannot use `el.annex.domain` since while it might be semantically equivalent, it does
    # not necessarily be equal.
    # can be reproduced in icon4py `dycore_tests/test_solve_nonhydro.py::test_nonhydro_predictor_step`
    # compute_theta_rho_face_values_and_pressure_gradient_and_update_vn
    assert isinstance(stmt.expr.annex.domain, tuple)
    assert domain in stmt.expr.annex.domain
    for i, (el, el_domain) in enumerate(zip(stmt.expr.args, stmt.expr.annex.domain)):
        assert not isinstance(el_domain, tuple), "Nested tuples not supported."
        assert isinstance(el_domain, SymbolicDomain)
        if el_domain == domain:
            new_targets.append(im.tuple_get(i, stmt.target))
            new_els.append(el)

    if len(new_els) == 0:
        breakpoint()

    new_expr = im.make_tuple(*new_els)
    new_expr.annex.domain = domain

    return itir.SetAt(expr=new_expr, domain=domain.as_expr(), target=im.make_tuple(*new_targets))


def _populate_and_homogenize_domains(stmts: list[itir.Stmt]) -> list[itir.Stmt]:
    new_stmts = []
    for stmt in stmts:
        if isinstance(stmt, itir.SetAt):
            assert hasattr(stmt.expr.annex, "domain")
            distinct_domains: set[domain_utils.SymbolicDomain] = set(
                # TODO: is flatten the right thing to do here? we don't supported nested tuples right?
                next_utils.flatten_nested_tuple(stmt.expr.annex.domain)  # type: ignore[assignment]  # mypy not smart enough
            )
            if len(distinct_domains) > 1:
                for domain in distinct_domains:
                    new_stmts.append(_filter_setat_by_domain(stmt, domain))
            else:
                new_stmts.append(
                    itir.SetAt(
                        expr=stmt.expr,
                        domain=next(iter(distinct_domains)).as_expr(),
                        target=stmt.target,
                    )
                )
        elif isinstance(stmt, itir.IfStmt):
            new_stmts.append(
                itir.IfStmt(
                    cond=stmt.cond,
                    true_branch=_populate_and_homogenize_domains(stmt.true_branch),
                    false_branch=_populate_and_homogenize_domains(stmt.false_branch),
                )
            )
        else:
            raise ValueError("Expected `itir.SetAt` or `itir.IfStmt`.")

    return new_stmts


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

            if not hasattr(tmp_expr.annex, "domain"):
                breakpoint()

            def get_domain(
                _, path: tuple[int, ...]
            ) -> domain_utils.SymbolicDomain:  # function only used inside loop
                domain = functools.reduce(
                    lambda var, idx: var[idx] if isinstance(var, tuple) else var,
                    path,
                    tmp_expr.annex.domain,
                )
                assert isinstance(domain, domain_utils.SymbolicDomain)
                return domain

            tmp_domains: (
                domain_utils.SymbolicDomain | tuple[domain_utils.SymbolicDomain | tuple, ...]
            ) = type_info.apply_to_primitive_constituents(
                get_domain,
                tmp_expr.type,
                with_path_arg=True,
                tuple_constructor=lambda *elements: tuple(elements),
            )

            # allocate temporary for all tuple elements
            def allocate_temporary(
                tmp_name: str, domain: domain_utils.SymbolicDomain, dtype: ts.ScalarType
            ):
                declarations.append(
                    itir.Temporary(id=tmp_name, domain=domain.as_expr(), dtype=dtype)
                )  # function only used inside loop

            next_utils.tree_map(allocate_temporary)(tmp_names, tmp_domains, tmp_dtypes)

            # if the expr is a field this just gives a simple `itir.SymRef`, otherwise we generate a
            #  `make_tuple` expression.
            def ref_with_domain(name: str, domain: domain_utils.SymbolicDomain) -> itir.SymRef:
                expr = im.ref(name)
                expr.annex.domain = domain
                return expr

            target_expr: itir.Expr = next_utils.tree_map(
                lambda name, domain: ref_with_domain(name, domain),
                result_collection_constructor=lambda els: im.make_tuple(*els),
            )(tmp_names, tmp_domains)  # type: ignore[assignment]  # typing of tree_map does not reflect action of `result_collection_constructor` yet

            # note: the let would be removed automatically by the `cse.extract_subexpression`, but
            # we remove it here for readability & debuggability.
            new_expr = inline_lambdas.inline_lambda(
                im.let(tmp_sym, target_expr)(new_expr), opcount_preserving=False
            )

            # TODO(tehrengruber): _transform_stmt not needed if deepest_expr_first=True
            tmp_stmts.extend(
                _transform_stmt(
                    itir.SetAt(
                        target=target_expr,
                        # the domain is populated later in `_populate_and_homogenize_domains`
                        domain=im.ref("UNPOPULATED"),
                        expr=tmp_expr,
                    ),
                    declarations,
                    uids,
                )
            )

        return [*tmp_stmts, itir.SetAt(target=stmt.target, domain=stmt.domain, expr=new_expr)]
    return None


def _transform_stmt(
    stmt: itir.Stmt, declarations: list[itir.Temporary], uids: eve_utils.UIDGenerator
) -> list[itir.Stmt]:
    unprocessed_stmts: list[itir.Stmt] = [stmt]
    stmts: list[itir.Stmt] = []

    transforms: list[Callable] = [
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

    stmts = _populate_and_homogenize_domains(stmts)

    return stmts


def create_global_tmps(
    program: itir.Program,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    #: A dictionary mapping axes names to their length. See :func:`infer_domain.infer_expr` for
    #: more details.
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    *,
    uids: Optional[eve_utils.UIDGenerator] = None,
) -> itir.Program:
    """
    Given an `itir.Program` create temporaries for intermediate values.

    This pass looks at all `as_fieldop` calls and transforms field-typed subexpressions of its
    arguments into temporaries.
    """
    offset_provider_type = common.offset_provider_to_type(offset_provider)
    program = infer_domain.infer_program(
        program, offset_provider=offset_provider, symbolic_domain_sizes=symbolic_domain_sizes
    )
    program = type_inference.infer(program, offset_provider_type=offset_provider_type)

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
