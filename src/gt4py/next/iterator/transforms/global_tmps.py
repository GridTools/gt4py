# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Callable, Literal, Optional, cast

from gt4py.eve import utils as eve_utils
from gt4py.next import common, utils as next_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as ir_utils_misc,
)
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicDomain
from gt4py.next.iterator.transforms import cse, infer_domain, inline_lambdas
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def select_elems_by_domain(
    select_domain: SymbolicDomain,
    target: itir.Expr,
    source: itir.Expr,
    domains: tuple[SymbolicDomain, ...],
):
    """
    Select all elements of possibly nested tuples for the given domain.

    Returns (non-nested) tuples of the selected elements and the corresponding targets.
    """
    new_targets = []
    new_els = []
    for i, el_domain in enumerate(domains):
        current_target = im.tuple_get(i, target)
        current_source = im.tuple_get(i, source)
        if isinstance(el_domain, tuple):
            more_targets, more_els = select_elems_by_domain(
                select_domain, current_target, current_source, el_domain
            )
            new_els.extend(more_els)
            new_targets.extend(more_targets)
        else:
            assert isinstance(el_domain, SymbolicDomain)
            if el_domain == select_domain:
                new_targets.append(current_target)
                new_els.append(current_source)
    return new_targets, new_els


def _set_at_for_domain(stmt: itir.SetAt, domain: SymbolicDomain) -> itir.SetAt:
    """Extract all elements with given domain into a new `SetAt` statement."""
    tuple_expr = stmt.expr
    targets, expr_els = select_elems_by_domain(
        domain, stmt.target, tuple_expr, stmt.expr.annex.domain
    )
    new_expr = im.make_tuple(*expr_els)
    new_expr.annex.domain = domain

    return itir.SetAt(expr=new_expr, domain=domain.as_expr(), target=im.make_tuple(*targets))


def _populate_and_homogenize_domains(stmts: list[itir.Stmt]) -> list[itir.Stmt]:
    """
    Splits `SetAt` statements with multiple domains into multiple statements.

    `SetAt`s have a single domain therefore the target domains have to be the same.
    We support nested tuples by flattening them and recombining them into non-nested tuples.
    """
    new_stmts: list[itir.Stmt] = []
    for stmt in stmts:
        if isinstance(stmt, itir.SetAt):
            assert hasattr(stmt.expr.annex, "domain")
            # ordered set for reproducibility
            domains: dict[
                SymbolicDomain | Literal[infer_domain.DomainAccessDescriptor.NEVER], None
            ] = dict.fromkeys(next_utils.flatten_nested_tuple(stmt.expr.annex.domain))
            # don't count NEVER as a different domain
            domains.pop(infer_domain.DomainAccessDescriptor.NEVER, None)
            distinct_domains: list[domain_utils.SymbolicDomain] = list(
                cast(dict[SymbolicDomain, None], domains)
            )
            if len(distinct_domains) == 1:
                new_stmts.append(
                    itir.SetAt(
                        expr=stmt.expr,
                        domain=distinct_domains[0].as_expr(),  # insert concrete domain
                        target=stmt.target,
                    )
                )
            else:
                new_stmts.extend(_set_at_for_domain(stmt, domain) for domain in distinct_domains)
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


def _is_as_fieldop_of_scan(expr: itir.Expr) -> bool:
    return (
        cpm.is_applied_as_fieldop(expr)
        and isinstance(expr.fun, itir.FunCall)
        and cpm.is_call_to(expr.fun.args[0], "scan")
    )


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

    # hide projector from extraction
    projector, expr = ir_utils_misc.extract_projector(stmt.expr)

    new_expr, extracted_fields, _ = cse.extract_subexpression(
        expr,
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
                tuple_constructor=lambda *elements: elements,
            )

            tmp_domains: SymbolicDomain | tuple[SymbolicDomain | tuple, ...] = tmp_expr.annex.domain

            if cpm.is_applied_as_fieldop(tmp_expr):
                # In this case all tuple elements have the same size (or will be `NEVER`).
                # Create the tuple structure with that domain.
                domain = list(
                    set(next_utils.flatten_nested_tuple((tmp_domains,)))
                    - {infer_domain.DomainAccessDescriptor.NEVER}  # type: ignore[arg-type] # type should always be `SymbolicDomain`
                )
                assert len(domain) == 1
                # this is the domain used as initial value in the tuple construction below
                tmp_domains = domain[0]

            def get_domain(
                _, path: tuple[int, ...]
            ) -> domain_utils.SymbolicDomain:  # function only used inside loop
                domain = functools.reduce(
                    lambda var, idx: var[idx] if isinstance(var, tuple) else var,
                    path,
                    tmp_domains,  # noqa: B023
                )
                assert isinstance(domain, domain_utils.SymbolicDomain)
                return domain

            # The following propagates the domains to the tuple structure of `tmp_expr.type`.
            # `tmp_domains` might not have this structure because domain inference was not able to infer the tuple structure.
            tmp_domains = type_info.apply_to_primitive_constituents(
                get_domain,
                tmp_expr.type,
                with_path_arg=True,
                tuple_constructor=lambda *elements: tuple(elements),
            )

            declarations.extend(
                itir.Temporary(id=tmp_name, domain=domain.as_expr(), dtype=dtype)
                for tmp_name, domain, dtype in zip(
                    next_utils.flatten_nested_tuple((tmp_names,)),
                    next_utils.flatten_nested_tuple((tmp_domains,)),
                    next_utils.flatten_nested_tuple((tmp_dtypes,)),
                    strict=True,
                )
            )

            # if the expr is a field this just gives a simple `itir.SymRef`, otherwise we generate a
            #  `make_tuple` expression.
            target_expr: itir.Expr = next_utils.tree_map(
                lambda name, domain: im.ref(name, annex={"domain": domain}),
                result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
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
                        # The domain is populated later in `_populate_and_homogenize_domains`
                        # at this point the `SetAt` contains possibly expressions on different domains.
                        domain=im.ref("UNPOPULATED"),
                        expr=tmp_expr,
                    ),
                    declarations,
                    uids,
                )
            )

        if projector is not None:
            # add the projector back
            domain = new_expr.annex.domain
            new_expr = im.call(projector)(new_expr)
            new_expr.annex.domain = domain

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
    program = infer_domain.infer_program(
        program,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        # Previous passes are allowed to create expressions without domains, but we must not
        # overwrite other domains here. Instead, only reinfer expressions without a domain. We must
        # not overwrite them since e.g. a `concat_where` expression might be rewritten into an
        # `if_(cond, tb, fb)` expression where re-inference might extend the domain of tb and fb.
        # See :class:`infer_domain.infer_expr` for details.
        keep_existing_domains=True,
    )
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
