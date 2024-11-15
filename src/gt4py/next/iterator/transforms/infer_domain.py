# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import typing
from typing import Callable, Optional, TypeAlias

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.utils import flatten_nested_tuple, tree_map


DOMAIN: TypeAlias = domain_utils.SymbolicDomain | None | tuple["DOMAIN", ...]
ACCESSED_DOMAINS: TypeAlias = dict[str, DOMAIN]


class DomainAnnexDebugger(eve.NodeVisitor):
    """
    Small utility class to debug missing domain attribute in annex.
    """

    def visit_Node(self, node: itir.Node):
        if cpm.is_applied_as_fieldop(node):
            if not hasattr(node.annex, "domain"):
                breakpoint()  # noqa: T100
        return self.generic_visit(node)


def _split_dict_by_key(pred: Callable, d: dict):
    """
    Split dictionary into two based on predicate.

    >>> d = {1: "a", 2: "b", 3: "c", 4: "d"}
    >>> _split_dict_by_key(lambda k: k % 2 == 0, d)
    ({2: 'b', 4: 'd'}, {1: 'a', 3: 'c'})
    """
    a: dict = {}
    b: dict = {}
    for k, v in d.items():
        (a if pred(k) else b)[k] = v
    return a, b


# TODO(tehrengruber): Revisit whether we want to move this behaviour to `domain_utils.domain_union`.
def _domain_union_with_none(
    *domains: domain_utils.SymbolicDomain | None,
) -> domain_utils.SymbolicDomain | None:
    filtered_domains: list[domain_utils.SymbolicDomain] = [d for d in domains if d is not None]
    if len(filtered_domains) == 0:
        return None
    return domain_utils.domain_union(*filtered_domains)


def _canonicalize_domain_structure(d1: DOMAIN, d2: DOMAIN) -> tuple[DOMAIN, DOMAIN]:
    """
    Given two domains or composites thereof, canonicalize their structure.

    If one of the arguments is a tuple the other one will be promoted to a tuple of same structure
    unless it already is a tuple. Missing values are replaced by None, meaning no domain is
    specified.

    >>> domain = im.domain(common.GridType.CARTESIAN, {})
    >>> _canonicalize_domain_structure((domain,), (domain, domain)) == (
    ...     (domain, None),
    ...     (domain, domain),
    ... )
    True

    >>> _canonicalize_domain_structure((domain, None), None) == ((domain, None), (None, None))
    True
    """
    if d1 is None and isinstance(d2, tuple):
        return _canonicalize_domain_structure((None,) * len(d2), d2)
    if d2 is None and isinstance(d1, tuple):
        return _canonicalize_domain_structure(d1, (None,) * len(d1))
    if isinstance(d1, tuple) and isinstance(d2, tuple):
        return tuple(
            zip(
                *(
                    _canonicalize_domain_structure(el1, el2)
                    for el1, el2 in itertools.zip_longest(d1, d2, fillvalue=None)
                )
            )
        )  # type: ignore[return-value]  # mypy not smart enough
    return d1, d2


def _merge_domains(
    original_domains: ACCESSED_DOMAINS,
    additional_domains: ACCESSED_DOMAINS,
) -> ACCESSED_DOMAINS:
    new_domains = {**original_domains}

    for key, domain in additional_domains.items():
        original_domain, domain = _canonicalize_domain_structure(
            original_domains.get(key, None), domain
        )
        new_domains[key] = tree_map(_domain_union_with_none)(original_domain, domain)

    return new_domains


def _extract_accessed_domains(
    stencil: itir.Expr,
    input_ids: list[str],
    target_domain: domain_utils.SymbolicDomain,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> ACCESSED_DOMAINS:
    accessed_domains: dict[str, domain_utils.SymbolicDomain | None] = {}

    shifts_results = trace_shifts.trace_stencil(stencil, num_args=len(input_ids))

    for in_field_id, shifts_list in zip(input_ids, shifts_results, strict=True):
        new_domains = [
            domain_utils.SymbolicDomain.translate(
                target_domain, shift, offset_provider, symbolic_domain_sizes
            )
            for shift in shifts_list
        ]
        # `None` means field is never accessed
        accessed_domains[in_field_id] = _domain_union_with_none(
            accessed_domains.get(in_field_id, None), *new_domains
        )

    return typing.cast(ACCESSED_DOMAINS, accessed_domains)


def _infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    target_domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.FunCall, ACCESSED_DOMAINS]:
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")
    if target_domain is None:
        raise ValueError("'target_domain' cannot be 'None'.")
    # FIXME[#1582](tehrengruber): Temporary solution for `tuple_get` on scan result. See `test_solve_triag`.
    if isinstance(target_domain, tuple):
        target_domain = _domain_union_with_none(*flatten_nested_tuple(target_domain))
    if not isinstance(target_domain, domain_utils.SymbolicDomain):
        raise ValueError("'target_domain' needs to be a 'domain_utils.SymbolicDomain'.")

    # `as_fieldop(stencil)(inputs...)`
    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    # ensure stencil has as many params as arguments
    assert not isinstance(stencil, itir.Lambda) or len(stencil.params) == len(applied_fieldop.args)

    input_ids: list[str] = []

    # Assign ids for all inputs to `as_fieldop`. `SymRef`s stay as is, nested `as_fieldop` get a
    # temporary id.
    tmp_uid_gen = eve_utils.UIDGenerator(prefix="__dom_inf")
    for in_field in inputs:
        if isinstance(in_field, itir.FunCall) or isinstance(in_field, itir.Literal):
            id_ = tmp_uid_gen.sequential_id()
        elif isinstance(in_field, itir.SymRef):
            id_ = in_field.id
        else:
            raise ValueError(f"Unsupported expression of type '{type(in_field)}'.")
        input_ids.append(id_)

    inputs_accessed_domains: ACCESSED_DOMAINS = _extract_accessed_domains(
        stencil, input_ids, target_domain, offset_provider, symbolic_domain_sizes
    )

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldop`s
    accessed_domains: ACCESSED_DOMAINS = {}
    transformed_inputs: list[itir.Expr] = []
    for in_field_id, in_field in zip(input_ids, inputs):
        transformed_input, accessed_domains_tmp = infer_expr(
            in_field, inputs_accessed_domains[in_field_id], offset_provider, symbolic_domain_sizes
        )
        transformed_inputs.append(transformed_input)

        accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)

    target_domain_expr = domain_utils.SymbolicDomain.as_expr(target_domain)
    transformed_call = im.as_fieldop(stencil, target_domain_expr)(*transformed_inputs)

    accessed_domains_without_tmp = {
        k: v
        for k, v in accessed_domains.items()
        if not k.startswith(tmp_uid_gen.prefix)  # type: ignore[arg-type] # prefix is always str
    }

    return transformed_call, accessed_domains_without_tmp


def _infer_let(
    let_expr: itir.FunCall,
    input_domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.FunCall, ACCESSED_DOMAINS]:
    assert cpm.is_let(let_expr)
    assert isinstance(let_expr.fun, itir.Lambda)  # just to make mypy happy
    transformed_calls_expr, accessed_domains = infer_expr(
        let_expr.fun.expr, input_domain, offset_provider, symbolic_domain_sizes
    )

    let_params = {param_sym.id for param_sym in let_expr.fun.params}
    accessed_domains_let_args, accessed_domains_outer = _split_dict_by_key(
        lambda k: k in let_params, accessed_domains
    )

    transformed_calls_args: list[itir.Expr] = []
    for param, arg in zip(let_expr.fun.params, let_expr.args, strict=True):
        transformed_calls_arg, accessed_domains_arg = infer_expr(
            arg,
            accessed_domains_let_args.get(
                param.id,
                None,
            ),
            offset_provider,
            symbolic_domain_sizes,
        )
        accessed_domains_outer = _merge_domains(accessed_domains_outer, accessed_domains_arg)
        transformed_calls_args.append(transformed_calls_arg)

    transformed_call = im.let(
        *(
            (str(param.id), call)
            for param, call in zip(let_expr.fun.params, transformed_calls_args, strict=True)
        )
    )(transformed_calls_expr)

    return transformed_call, accessed_domains_outer


def _infer_make_tuple(
    expr: itir.Expr,
    domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    assert cpm.is_call_to(expr, "make_tuple")
    infered_args_expr = []
    actual_domains: ACCESSED_DOMAINS = {}
    if not isinstance(domain, tuple):
        # promote domain to a tuple of domains such that it has the same structure as
        # the expression
        # TODO(tehrengruber): Revisit. Still open how to handle IR in this case example:
        #  out @ c⟨ IDimₕ: [0, __out_size_0) ⟩ ← {__sym_1, __sym_2};
        domain = (domain,) * len(expr.args)
    assert len(expr.args) >= len(domain)
    # There may be less domains than tuple args, pad the domain with `None` in that case.
    #  e.g. `im.tuple_get(0, im.make_tuple(a, b), domain=domain)`
    domain = (*domain, *(None for _ in range(len(expr.args) - len(domain))))
    for i, arg in enumerate(expr.args):
        infered_arg_expr, actual_domains_arg = infer_expr(
            arg, domain[i], offset_provider, symbolic_domain_sizes
        )
        infered_args_expr.append(infered_arg_expr)
        actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    result_expr = im.call(expr.fun)(*infered_args_expr)
    return result_expr, actual_domains


def _infer_tuple_get(
    expr: itir.Expr,
    domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    assert cpm.is_call_to(expr, "tuple_get")
    actual_domains: ACCESSED_DOMAINS = {}
    idx_expr, tuple_arg = expr.args
    assert isinstance(idx_expr, itir.Literal)
    idx = int(idx_expr.value)
    tuple_domain = tuple(None if i != idx else domain for i in range(idx + 1))
    infered_arg_expr, actual_domains_arg = infer_expr(
        tuple_arg, tuple_domain, offset_provider, symbolic_domain_sizes
    )

    infered_args_expr = im.tuple_get(idx, infered_arg_expr)
    actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    return infered_args_expr, actual_domains


def _infer_if(
    expr: itir.Expr,
    domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    assert cpm.is_call_to(expr, "if_")
    infered_args_expr = []
    actual_domains: ACCESSED_DOMAINS = {}
    cond, true_val, false_val = expr.args
    for arg in [true_val, false_val]:
        infered_arg_expr, actual_domains_arg = infer_expr(
            arg, domain, offset_provider, symbolic_domain_sizes
        )
        infered_args_expr.append(infered_arg_expr)
        actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    result_expr = im.call(expr.fun)(cond, *infered_args_expr)
    return result_expr, actual_domains


def _infer_expr(
    expr: itir.Expr,
    domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    if isinstance(expr, itir.SymRef):
        return expr, {str(expr.id): domain}
    elif isinstance(expr, itir.Literal):
        return expr, {}
    elif cpm.is_applied_as_fieldop(expr):
        return _infer_as_fieldop(expr, domain, offset_provider, symbolic_domain_sizes)
    elif cpm.is_let(expr):
        return _infer_let(expr, domain, offset_provider, symbolic_domain_sizes)
    elif cpm.is_call_to(expr, "make_tuple"):
        return _infer_make_tuple(expr, domain, offset_provider, symbolic_domain_sizes)
    elif cpm.is_call_to(expr, "tuple_get"):
        return _infer_tuple_get(expr, domain, offset_provider, symbolic_domain_sizes)
    elif cpm.is_call_to(expr, "if_"):
        return _infer_if(expr, domain, offset_provider, symbolic_domain_sizes)
    elif (
        cpm.is_call_to(expr, itir.ARITHMETIC_BUILTINS)
        or cpm.is_call_to(expr, itir.TYPEBUILTINS)
        or cpm.is_call_to(expr, ("cast_", "index", "unstructured_domain", "cartesian_domain"))
    ):
        return expr, {}
    else:
        raise ValueError(f"Unsupported expression: {expr}")


def infer_expr(
    expr: itir.Expr,
    domain: DOMAIN,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    """
    Infer the domain of all field subexpressions of `expr`.

    Given an expression `expr` and the domain it is accessed at, back-propagate the domain of all
    (field-typed) subexpression.

    Arguments:
    - expr: The expression to be inferred.
    - domain: The domain `expr` is read at.
    - symbolic_domain_sizes: A dictionary mapping axes names, e.g., `I`, `Vertex`, to a symbol
      name that evaluates to the length of that axis.

    Returns:
      A tuple containing the inferred expression with all applied `as_fieldop` (that are accessed)
      having a domain argument now, and a dictionary mapping symbol names referenced in `expr` to
      domain they are accessed at.
    """
    # this is just a small wrapper that populates the `domain` annex
    expr, accessed_domains = _infer_expr(expr, domain, offset_provider, symbolic_domain_sizes)
    expr.annex.domain = domain
    return expr, accessed_domains


def _infer_stmt(
    stmt: itir.Stmt,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
):
    if isinstance(stmt, itir.SetAt):
        transformed_call, _unused_domain = infer_expr(
            stmt.expr,
            domain_utils.SymbolicDomain.from_expr(stmt.domain),
            offset_provider,
            symbolic_domain_sizes,
        )
        return itir.SetAt(
            expr=transformed_call,
            domain=stmt.domain,
            target=stmt.target,
        )
    elif isinstance(stmt, itir.IfStmt):
        return itir.IfStmt(
            cond=stmt.cond,
            true_branch=[
                _infer_stmt(c, offset_provider, symbolic_domain_sizes) for c in stmt.true_branch
            ],
            false_branch=[
                _infer_stmt(c, offset_provider, symbolic_domain_sizes) for c in stmt.false_branch
            ],
        )
    raise ValueError(f"Unsupported stmt: {stmt}")


def infer_program(
    program: itir.Program,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
) -> itir.Program:
    """
    Infer the domain of all field subexpressions inside a program.

    See :func:`infer_expr` for more details.
    """
    assert (
        not program.function_definitions
    ), "Domain propagation does not support function definitions."

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=program.declarations,
        body=[_infer_stmt(stmt, offset_provider, symbolic_domain_sizes) for stmt in program.body],
    )
