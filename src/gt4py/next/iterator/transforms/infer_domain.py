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

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Callable, Optional, TypeAlias, Unpack
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.utils import flatten_nested_tuple, tree_map


class DomainAccessDescriptor(eve.StrEnum):
    """
    Descriptor for domains that could not be inferred.
    """

    # TODO(tehrengruber): Revisit this concept. It is strange that we don't have a descriptor
    #  `KNOWN`, but since we don't need it, it wasn't added.

    #: The access is unknown because of a dynamic shift.whose extent is not known.
    #: E.g.: `(⇑(λ(arg0, arg1) → ·⟪Ioffₒ, ·arg1⟫(arg0)))(in_field1, in_field2)`
    UNKNOWN = "unknown"
    #: The domain is never accessed.
    #: E.g.: `{in_field1, in_field2}[0]`
    NEVER = "never"


NonTupleDomainAccess: TypeAlias = domain_utils.SymbolicDomain | DomainAccessDescriptor
#: The domain can also be a tuple of domains, usually this only occurs for scan operators returning
#: a tuple since other occurrences for tuples are removed before domain inference. This is
#: however not a requirement of the pass and `make_tuple(vertex_field, edge_field)` infers just
#: fine to a tuple of a vertex and an edge domain.
DomainAccess: TypeAlias = NonTupleDomainAccess | tuple["DomainAccess", ...]
AccessedDomains: TypeAlias = dict[str, DomainAccess]


class InferenceOptions(typing.TypedDict):
    offset_provider: common.OffsetProvider
    symbolic_domain_sizes: Optional[dict[str, str]]
    allow_uninferred: bool


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
def _domain_union(
    *domains: domain_utils.SymbolicDomain | DomainAccessDescriptor,
) -> domain_utils.SymbolicDomain | DomainAccessDescriptor:
    if any(d == DomainAccessDescriptor.UNKNOWN for d in domains):
        return DomainAccessDescriptor.UNKNOWN

    filtered_domains: list[domain_utils.SymbolicDomain] = [
        d  # type: ignore[misc]  # domain can never be unknown as these cases are filtered above
        for d in domains
        if d != DomainAccessDescriptor.NEVER
    ]
    if len(filtered_domains) == 0:
        return DomainAccessDescriptor.NEVER
    return domain_utils.domain_union(*filtered_domains)


def _canonicalize_domain_structure(
    d1: DomainAccess, d2: DomainAccess
) -> tuple[DomainAccess, DomainAccess]:
    """
    Given two domains or composites thereof, canonicalize their structure.

    If one of the arguments is a tuple the other one will be promoted to a tuple of same structure
    unless it already is a tuple. Missing values are filled by :ref:`DomainAccessDescriptor.NEVER`.

    >>> domain = im.domain(common.GridType.CARTESIAN, {})
    >>> _canonicalize_domain_structure((domain,), (domain, domain)) == (
    ...     (domain, DomainAccessDescriptor.NEVER),
    ...     (domain, domain),
    ... )
    True

    >>> _canonicalize_domain_structure(
    ...     (domain, DomainAccessDescriptor.NEVER), DomainAccessDescriptor.NEVER
    ... ) == (
    ...     (domain, DomainAccessDescriptor.NEVER),
    ...     (DomainAccessDescriptor.NEVER, DomainAccessDescriptor.NEVER),
    ... )
    True
    """
    if d1 is DomainAccessDescriptor.NEVER and isinstance(d2, tuple):
        return _canonicalize_domain_structure((DomainAccessDescriptor.NEVER,) * len(d2), d2)
    if d2 is DomainAccessDescriptor.NEVER and isinstance(d1, tuple):
        return _canonicalize_domain_structure(d1, (DomainAccessDescriptor.NEVER,) * len(d1))
    if isinstance(d1, tuple) and isinstance(d2, tuple):
        return tuple(
            zip(
                *(
                    _canonicalize_domain_structure(el1, el2)
                    for el1, el2 in itertools.zip_longest(
                        d1, d2, fillvalue=DomainAccessDescriptor.NEVER
                    )
                )
            )
        )  # type: ignore[return-value]  # mypy not smart enough
    return d1, d2


def _merge_domains(
    original_domains: AccessedDomains,
    additional_domains: AccessedDomains,
) -> AccessedDomains:
    new_domains = {**original_domains}

    for key, domain in additional_domains.items():
        original_domain, domain = _canonicalize_domain_structure(
            original_domains.get(key, DomainAccessDescriptor.NEVER), domain
        )
        new_domains[key] = tree_map(_domain_union)(original_domain, domain)

    return new_domains


def _extract_accessed_domains(
    stencil: itir.Expr,
    input_ids: list[str],
    input_types: list[Optional[ts.TypeSpec]],
    target_domain: NonTupleDomainAccess,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
) -> dict[str, NonTupleDomainAccess]:
    accessed_domains: dict[str, NonTupleDomainAccess] = {}

    shifts_results = trace_shifts.trace_stencil(stencil, num_args=len(input_ids))

    for input_id, input_type, shifts_list in zip(
        input_ids, input_types, shifts_results, strict=True
    ):
        if isinstance(
            input_type, ts.ScalarType
        ):  # TODO: only do loop body when field instead (requires complete type information)
            continue
        # TODO(tehrengruber): Dynamic shifts are not supported by `SymbolicDomain.translate`. Use
        #  special `UNKNOWN` marker for them until we have implemented a proper solution.
        if any(s == trace_shifts.Sentinel.VALUE for shift in shifts_list for s in shift):
            accessed_domains[input_id] = DomainAccessDescriptor.UNKNOWN
            continue

        new_domains = [
            domain_utils.SymbolicDomain.translate(
                target_domain, shift, offset_provider, symbolic_domain_sizes
            )
            if not isinstance(target_domain, DomainAccessDescriptor)
            else target_domain
            for shift in shifts_list
        ]
        accessed_domains[input_id] = _domain_union(
            accessed_domains.get(input_id, DomainAccessDescriptor.NEVER), *new_domains
        )

    return accessed_domains


def _infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    target_domain: DomainAccess,
    *,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]],
    allow_uninferred: bool,
) -> tuple[itir.FunCall, AccessedDomains]:
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")
    if not allow_uninferred and target_domain is DomainAccessDescriptor.NEVER:
        raise ValueError("'target_domain' cannot be 'NEVER' unless `allow_uninferred=True`.")
    # FIXME[#1582](tehrengruber): Temporary solution for `tuple_get` on scan result. See `test_solve_triag`.
    if isinstance(target_domain, tuple):
        target_domain = _domain_union(*flatten_nested_tuple(target_domain))  # type: ignore[arg-type]  # mypy not smart enough
    assert isinstance(target_domain, (domain_utils.SymbolicDomain, DomainAccessDescriptor))

    # `as_fieldop(stencil)(inputs...)`
    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    # ensure stencil has as many params as arguments
    assert not isinstance(stencil, itir.Lambda) or len(stencil.params) == len(applied_fieldop.args)

    input_ids: list[str] = []
    input_types: list[Optional[ts.TypeSpec]] = []

    # Assign ids for all inputs to `as_fieldop`. `SymRef`s stay as is, nested `as_fieldop` get a
    # temporary id.
    tmp_uid_gen = eve_utils.UIDGenerator(prefix="__dom_inf")
    for input in inputs:
        if isinstance(input, itir.FunCall) or isinstance(input, itir.Literal):
            id_ = tmp_uid_gen.sequential_id()
        elif isinstance(input, itir.SymRef):
            id_ = input.id
        else:
            raise ValueError(f"Unsupported expression of type '{type(input)}'.")
        input_ids.append(id_)
        input_types.append(input.type)

    inputs_accessed_domains: dict[str, NonTupleDomainAccess] = _extract_accessed_domains(
        stencil, input_ids, input_types, target_domain, offset_provider, symbolic_domain_sizes
    )

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldop`s
    accessed_domains: AccessedDomains = {}
    transformed_inputs: list[itir.Expr] = []
    for input_id, input_type, input in zip(input_ids, input_types, inputs):
        if isinstance(
            input_type, ts.ScalarType
        ):  # TODO: only do loop body when field instead (requires complete type information)
            transformed_inputs.append(input)
        else:
            transformed_input, accessed_domains_tmp = infer_expr(
                input,
                inputs_accessed_domains[input_id],
                offset_provider=offset_provider,
                symbolic_domain_sizes=symbolic_domain_sizes,
                allow_uninferred=allow_uninferred,
            )
            transformed_inputs.append(transformed_input)

            accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)

    if not isinstance(target_domain, DomainAccessDescriptor):
        target_domain_expr = domain_utils.SymbolicDomain.as_expr(target_domain)
    else:
        target_domain_expr = None
    transformed_call = im.as_fieldop(stencil, target_domain_expr)(*transformed_inputs)

    accessed_domains_without_tmp = {
        k: v
        for k, v in accessed_domains.items()
        if not k.startswith(tmp_uid_gen.prefix)  # type: ignore[arg-type] # prefix is always str
    }

    return transformed_call, accessed_domains_without_tmp


def _infer_let(
    let_expr: itir.FunCall,
    input_domain: DomainAccess,
    **kwargs: Unpack[InferenceOptions],
) -> tuple[itir.FunCall, AccessedDomains]:
    assert cpm.is_let(let_expr)
    assert isinstance(let_expr.fun, itir.Lambda)  # just to make mypy happy
    let_params = {param_sym.id for param_sym in let_expr.fun.params}

    transformed_calls_expr, accessed_domains = infer_expr(let_expr.fun.expr, input_domain, **kwargs)

    accessed_domains_let_args, accessed_domains_outer = _split_dict_by_key(
        lambda k: k in let_params, accessed_domains
    )

    transformed_calls_args: list[itir.Expr] = []
    for param, arg in zip(let_expr.fun.params, let_expr.args, strict=True):
        if isinstance(
            arg.type, ts.ScalarType
        ):  # TODO: only do loop body when field instead (requires complete type information)
            transformed_calls_arg = arg
        else:
            transformed_calls_arg, accessed_domains_arg = infer_expr(
                arg,
                accessed_domains_let_args.get(
                    param.id,
                    DomainAccessDescriptor.NEVER,
                ),
                **kwargs,
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
    domain: DomainAccess,
    **kwargs: Unpack[InferenceOptions],
) -> tuple[itir.Expr, AccessedDomains]:
    assert cpm.is_call_to(expr, "make_tuple")
    infered_args_expr = []
    actual_domains: AccessedDomains = {}
    if not isinstance(domain, tuple):
        # promote domain to a tuple of domains such that it has the same structure as
        # the expression
        # TODO(tehrengruber): Revisit. Still open how to handle IR in this case example:
        #  out @ c⟨ IDimₕ: [0, __out_size_0) ⟩ ← {__sym_1, __sym_2};
        domain = (domain,) * len(expr.args)
    assert len(expr.args) >= len(domain)
    # There may be fewer domains than tuple args, pad the domain with `NEVER`
    # in that case.
    # e.g. `im.tuple_get(0, im.make_tuple(a, b), domain=domain)`
    domain = (*domain, *(DomainAccessDescriptor.NEVER for _ in range(len(expr.args) - len(domain))))
    for i, arg in enumerate(expr.args):
        infered_arg_expr, actual_domains_arg = infer_expr(arg, domain[i], **kwargs)
        infered_args_expr.append(infered_arg_expr)
        actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    result_expr = im.call(expr.fun)(*infered_args_expr)
    return result_expr, actual_domains


def _infer_tuple_get(
    expr: itir.Expr,
    domain: DomainAccess,
    **kwargs: Unpack[InferenceOptions],
) -> tuple[itir.Expr, AccessedDomains]:
    assert cpm.is_call_to(expr, "tuple_get")
    actual_domains: AccessedDomains = {}
    idx_expr, tuple_arg = expr.args
    assert isinstance(idx_expr, itir.Literal)
    idx = int(idx_expr.value)
    tuple_domain = tuple(
        DomainAccessDescriptor.NEVER if i != idx else domain for i in range(idx + 1)
    )
    infered_arg_expr, actual_domains_arg = infer_expr(tuple_arg, tuple_domain, **kwargs)

    infered_args_expr = im.tuple_get(idx, infered_arg_expr)
    actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    return infered_args_expr, actual_domains


def _infer_if(
    expr: itir.Expr,
    domain: DomainAccess,
    **kwargs: Unpack[InferenceOptions],
) -> tuple[itir.Expr, AccessedDomains]:
    assert cpm.is_call_to(expr, "if_")
    infered_args_expr = []
    actual_domains: AccessedDomains = {}
    cond, true_val, false_val = expr.args
    for arg in [true_val, false_val]:
        infered_arg_expr, actual_domains_arg = infer_expr(arg, domain, **kwargs)
        infered_args_expr.append(infered_arg_expr)
        actual_domains = _merge_domains(actual_domains, actual_domains_arg)
    result_expr = im.call(expr.fun)(cond, *infered_args_expr)
    return result_expr, actual_domains


def _infer_expr(
    expr: itir.Expr,
    domain: DomainAccess,
    **kwargs: Unpack[InferenceOptions],
) -> tuple[itir.Expr, AccessedDomains]:
    if isinstance(expr, itir.SymRef):
        return expr, {str(expr.id): domain}
    elif isinstance(expr, itir.Literal):
        return expr, {}
    elif cpm.is_applied_as_fieldop(expr):
        return _infer_as_fieldop(expr, domain, **kwargs)
    elif cpm.is_let(expr):
        return _infer_let(expr, domain, **kwargs)
    elif cpm.is_call_to(expr, "make_tuple"):
        return _infer_make_tuple(expr, domain, **kwargs)
    elif cpm.is_call_to(expr, "tuple_get"):
        return _infer_tuple_get(expr, domain, **kwargs)
    elif cpm.is_call_to(expr, "if_"):
        return _infer_if(expr, domain, **kwargs)
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
    domain: DomainAccess,
    *,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    allow_uninferred: bool = False,
) -> tuple[itir.Expr, AccessedDomains]:
    """
    Infer the domain of all field subexpressions of `expr`.

    Given an expression `expr` and the domain it is accessed at, back-propagate the domain of all
    (field-typed) subexpression.

    Arguments:
    - expr: The expression to be inferred.
    - domain: The domain `expr` is read at.
    - symbolic_domain_sizes: A dictionary mapping axes names, e.g., `I`, `Vertex`, to a symbol
      name that evaluates to the length of that axis.
    - allow_uninferred: Allow `as_fieldop` expressions whose domain is either unknown (e.g.
      because of a dynamic shift) or never accessed.

    Returns:
      A tuple containing the inferred expression with all applied `as_fieldop` (that are accessed)
      having a domain argument now, and a dictionary mapping symbol names referenced in `expr` to
      domain they are accessed at.
    """
    expr, accessed_domains = _infer_expr(
        expr,
        domain,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        allow_uninferred=allow_uninferred,
    )
    expr.annex.domain = domain

    return expr, accessed_domains


def _infer_stmt(
    stmt: itir.Stmt,
    **kwargs: Unpack[InferenceOptions],
):
    if isinstance(stmt, itir.SetAt):
        transformed_call, _ = infer_expr(
            stmt.expr, domain_utils.SymbolicDomain.from_expr(stmt.domain), **kwargs
        )

        return itir.SetAt(
            expr=transformed_call,
            domain=stmt.domain,
            target=stmt.target,
        )
    elif isinstance(stmt, itir.IfStmt):
        return itir.IfStmt(
            cond=stmt.cond,
            true_branch=[_infer_stmt(c, **kwargs) for c in stmt.true_branch],
            false_branch=[_infer_stmt(c, **kwargs) for c in stmt.false_branch],
        )
    raise ValueError(f"Unsupported stmt: {stmt}")


def infer_program(
    program: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    allow_uninferred: bool = False,
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
        body=[
            _infer_stmt(
                stmt,
                offset_provider=offset_provider,
                symbolic_domain_sizes=symbolic_domain_sizes,
                allow_uninferred=allow_uninferred,
            )
            for stmt in program.body
        ],
    )
