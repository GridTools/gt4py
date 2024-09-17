# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import itertools
from typing import Callable, TypeAlias

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, domain_union
from gt4py.next.iterator.transforms import trace_shifts


DOMAIN_TYPES: TypeAlias = SymbolicDomain | None
ACCESSED_DOMAINS: TypeAlias = dict[str, DOMAIN_TYPES | tuple[DOMAIN_TYPES]]


def split_dict_by_key(pred: Callable, d: dict):
    """
    Split dictionary into two based on predicate.

    >>> d = {1: "a", 2: "b", 3: "c", 4: "d"}
    >>> split_dict_by_key(lambda k: k % 2 == 0, d)
    ({2: 'b', 4: 'd'}, {1: 'a', 3: 'c'})
    """
    a: dict = {}
    b: dict = {}
    for k, v in d.items():
        (a if pred(k) else b)[k] = v
    return a, b


def _merge_domains(
    original_domains: ACCESSED_DOMAINS,
    additional_domains: ACCESSED_DOMAINS,
) -> ACCESSED_DOMAINS:
    new_domains = {**original_domains}

    for key, domain in additional_domains.items():
        original_domain = original_domains.get(key)

        if original_domain is None:
            new_domains[key] = domain
        elif domain is None:
            new_domains[key] = original_domain
        elif not isinstance(domain, tuple) and not isinstance(original_domain, tuple):
            new_domains[key] = domain_union([original_domain, domain])
        elif isinstance(domain, tuple) and isinstance(original_domain, tuple):
            new_domains[key] = tuple(
                domain_union([d1, d2])
                if d1 is not None and d2 is not None
                else (d1 if d2 is None else d2)
                for d1, d2 in itertools.zip_longest(original_domain, domain, fillvalue=None)
            )
        elif isinstance(domain, tuple):
            new_domains[key] = tuple(
                domain_union([d, original_domain]) if d is not None else original_domain
                for d in domain
            )
        elif isinstance(original_domain, tuple):
            new_domains[key] = tuple(
                domain_union([domain, d]) if d is not None else domain for d in original_domain
            )

    return new_domains


def extract_shifts_and_translate_domains(
    stencil: itir.Expr,
    input_ids: list[str],
    target_domain: SymbolicDomain,
    offset_provider: common.OffsetProvider,
    accessed_domains: ACCESSED_DOMAINS,
):
    shifts_results = trace_shifts.trace_stencil(stencil, num_args=len(input_ids))

    for in_field_id, shifts_list in zip(input_ids, shifts_results, strict=True):
        new_domains = [
            SymbolicDomain.translate(target_domain, shift, offset_provider) for shift in shifts_list
        ]
        if new_domains:
            accessed_domains[in_field_id] = domain_union(new_domains)
        else:
            accessed_domains.setdefault(in_field_id, None)


def infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    target_domain: SymbolicDomain | None,
    offset_provider: common.OffsetProvider,
) -> tuple[itir.FunCall, ACCESSED_DOMAINS]:
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")
    if target_domain is None:
        raise ValueError("'target_domain' cannot be 'None'.")

    # `as_fieldop(stencil)(inputs...)`
    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    # ensure stencil has as many params as arguments
    assert not isinstance(stencil, itir.Lambda) or len(stencil.params) == len(applied_fieldop.args)

    input_ids: list[str] = []
    accessed_domains: ACCESSED_DOMAINS = {}

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

    extract_shifts_and_translate_domains(
        stencil, input_ids, target_domain, offset_provider, accessed_domains
    )

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldops`
    transformed_inputs: list[itir.Expr] = []
    for in_field_id, in_field in zip(input_ids, inputs):
        if in_field_id not in accessed_domains:
            raise ValueError(
                f"Let param `{in_field_id}` is never accessed. Can not infer its domain."
            )
        transformed_input, accessed_domains_tmp = infer_expr(
            in_field, accessed_domains[in_field_id], offset_provider
        )
        transformed_inputs.append(transformed_input)

        # Merge accessed_domains and accessed_domains_tmp
        accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)

    transformed_call = im.as_fieldop(stencil, SymbolicDomain.as_expr(target_domain))(
        *transformed_inputs
    )

    accessed_domains_without_tmp = {
        k: v
        for k, v in accessed_domains.items()
        if not k.startswith(tmp_uid_gen.prefix)  # type: ignore[arg-type] # prefix is always str
    }

    return transformed_call, accessed_domains_without_tmp


def infer_let(
    let_expr: itir.FunCall,
    input_domain: DOMAIN_TYPES | tuple[DOMAIN_TYPES],
    offset_provider: common.OffsetProvider,
) -> tuple[itir.FunCall, ACCESSED_DOMAINS]:
    assert cpm.is_let(let_expr)
    assert isinstance(let_expr.fun, itir.Lambda)
    transformed_calls_expr, accessed_domains = infer_expr(
        let_expr.fun.expr, input_domain, offset_provider
    )

    # TODO(tehrengruber): describe and tidy up
    let_params = {param_sym.id for param_sym in let_expr.fun.params}
    accessed_domains_let_args, accessed_domains_outer = split_dict_by_key(
        lambda k: k in let_params, accessed_domains
    )

    transformed_calls_args: list[itir.Expr] = []
    for param, arg in zip(let_expr.fun.params, let_expr.args):
        transformed_calls_arg, accessed_domains_arg = infer_expr(
            arg,
            accessed_domains_let_args.get(
                param.id,
                None
                if (not isinstance(arg, tuple) and not cpm.is_call_to(arg, "make_tuple"))
                else tuple(None for _ in arg.args),
            ),
            offset_provider,
        )
        accessed_domains_outer = _merge_domains(accessed_domains_outer, accessed_domains_arg)
        transformed_calls_args.append(transformed_calls_arg)

    transformed_call = im.let(
        *((str(param.id), call) for param, call in zip(let_expr.fun.params, transformed_calls_args))
    )(transformed_calls_expr)

    return transformed_call, accessed_domains_outer


def infer_expr(
    expr: itir.Expr,
    domain: DOMAIN_TYPES | tuple[DOMAIN_TYPES],
    offset_provider: common.OffsetProvider,
) -> tuple[itir.Expr, ACCESSED_DOMAINS]:
    if isinstance(expr, itir.SymRef):
        return expr, {str(expr.id): domain}
    elif isinstance(expr, itir.Literal):
        return expr, {}
    elif cpm.is_applied_as_fieldop(expr):
        return infer_as_fieldop(expr, domain, offset_provider)
    elif cpm.is_let(expr):
        return infer_let(expr, domain, offset_provider)
    elif cpm.is_call_to(expr, itir.GTIR_BUILTINS):
        infered_args_expr = []
        actual_domains: ACCESSED_DOMAINS = {}
        if cpm.is_call_to(expr, "make_tuple"):
            if not isinstance(domain, tuple):
                domain = (
                    (domain,) * (len(expr.args))
                )  # TODO: Still open how to handle IR in this case, example: out @ c⟨ IDimₕ: [0, __out_size_0) ⟩ ← {__sym_1, __sym_2};
            if (
                not len(domain) <= len(expr.args)
            ):  # There may be less domains than tuple args, e.g. in im.tuple_get(0, im.make_tuple(a, b), domain=domain)
                raise ValueError("infer_expr needs a domain for each tuple element.")
            # This is to pad the domain length to the length of the tuple
            domain = (*domain, *(None for _ in range(len(expr.args) - len(domain))))
            for i, arg in enumerate(expr.args):
                infered_arg_expr, actual_domains_arg = infer_expr(arg, domain[i], offset_provider)
                infered_args_expr.append(infered_arg_expr)
                actual_domains = _merge_domains(actual_domains, actual_domains_arg)
            return im.call(expr.fun)(*infered_args_expr), actual_domains
        elif cpm.is_call_to(expr, "tuple_get"):
            idx, tuple_arg = expr.args
            assert isinstance(idx, itir.Literal)
            child_domain = tuple(
                None if i != int(idx.value) else domain for i in range(int(idx.value) + 1)
            )
            infered_arg_expr, actual_domains_arg = infer_expr(
                tuple_arg, child_domain, offset_provider
            )

            infered_args_expr = im.tuple_get(idx.value, infered_arg_expr)
            actual_domains = _merge_domains(actual_domains, actual_domains_arg)
            return infered_args_expr, actual_domains
        elif cpm.is_call_to(expr, "cond"):
            cond, true_val, false_val = expr.args
            for arg in [true_val, false_val]:
                infered_arg_expr, actual_domains_arg = infer_expr(arg, domain, offset_provider)
                infered_args_expr.append(infered_arg_expr)
                actual_domains = _merge_domains(actual_domains, actual_domains_arg)
            return im.call(expr.fun)(cond, *infered_args_expr), actual_domains
        elif (
            cpm.is_call_to(expr, itir.ARITHMETIC_BUILTINS)
            or cpm.is_call_to(expr, itir.TYPEBUILTINS)
            or cpm.is_call_to(expr, "cast_")
        ):
            return expr, actual_domains
        else:
            raise ValueError(f"Unsupported expression: {expr}")
    else:
        raise ValueError(f"Unsupported expression: {expr}")


def infer_program(
    program: itir.Program,
    offset_provider: dict[str, Dimension],
) -> itir.Program:
    transformed_set_ats: list[itir.SetAt] = []
    assert not program.function_definitions  # TODO(tehrengruber): error message

    for set_at in program.body:
        assert isinstance(set_at, itir.SetAt)
        if isinstance(set_at.expr, itir.SymRef):
            transformed_set_ats.append(set_at)
            continue
        assert isinstance(set_at.expr, itir.Expr)
        assert isinstance(
            set_at.target, itir.SymRef
        )  # TODO: stmt.target can be an expr, e.g. make_tuple

        transformed_call, _unused_domain = infer_expr(
            set_at.expr, SymbolicDomain.from_expr(set_at.domain), offset_provider
        )
        transformed_set_ats.append(
            itir.SetAt(
                expr=transformed_call,
                domain=(set_at.domain),
                target=set_at.target,
            ),
        )

    new_declarations = copy.deepcopy(program.declarations)

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=new_declarations,
        body=transformed_set_ats,
    )
