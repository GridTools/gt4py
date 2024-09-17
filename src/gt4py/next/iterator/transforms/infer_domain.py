# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Dict, Tuple
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import AUTO_DOMAIN, SymbolicDomain, domain_union
from gt4py.next.iterator.transforms import trace_shifts


def _merge_domains(
    original_domains: Dict[str, SymbolicDomain], additional_domains: Dict[str, SymbolicDomain]
) -> Dict[str, SymbolicDomain]:
    new_domains = {**original_domains}
    for key, value in additional_domains.items():
        if key in original_domains:
            new_domains[key] = domain_union([original_domains[key], value])
        else:
            new_domains[key] = value

    return new_domains

def extract_shifts_and_translate_domains(
    stencil: itir.Expr,
    input_ids: list[str],
    target_domain: SymbolicDomain,
    offset_provider: Dict[str, Dimension],
    accessed_domains: Dict[str, SymbolicDomain],
):
    shifts_results = trace_shifts.trace_stencil(
        stencil,
        num_args=len(input_ids)
    )

    for in_field_id, shifts_list in zip(input_ids, shifts_results, strict=True):
        new_domains = [
            SymbolicDomain.translate(target_domain, shift, offset_provider) for shift in shifts_list
        ]
        if new_domains:
            accessed_domains[in_field_id] = domain_union(new_domains)


def infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    target_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")

    # `as_fieldop(stencil)(inputs...)`
    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    # ensure stencil has as many params as arguments
    assert not isinstance(stencil, itir.Lambda) or len(stencil.params) == len(applied_fieldop.args)

    input_ids: list[str] = []
    accessed_domains: Dict[str, SymbolicDomain] = {}

    # Assign ids for all inputs to `as_fieldop`. `SymRef`s stay as is, nested `as_fieldop` get a
    # temporary id.
    tmp_uid_gen = eve_utils.UIDGenerator(prefix="__dom_inf")
    for in_field in inputs:
        if isinstance(in_field, itir.FunCall) or isinstance(in_field, itir.Literal):
            id_ = tmp_uid_gen.sequential_id()
        elif isinstance(in_field, itir.SymRef):
            id_ = in_field.id
        else:
            raise ValueError(f"Unsupported type {type(in_field)}")
        input_ids.append(id_)

    if isinstance(target_domain, itir.FunCall):
        target_domain = SymbolicDomain.from_expr(target_domain)

    extract_shifts_and_translate_domains(
        stencil, input_ids, target_domain, offset_provider, accessed_domains
    )

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldops`
    transformed_inputs: list[itir.Expr] = []
    for in_field_id, in_field in zip(input_ids, inputs):
        if isinstance(in_field, itir.FunCall):
            transformed_input, accessed_domains_tmp = infer_as_fieldop(
                in_field, accessed_domains[in_field_id], offset_provider
            )
            transformed_inputs.append(transformed_input)

            # Merge accessed_domains and accessed_domains_tmp
            accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)
        elif isinstance(in_field, itir.SymRef) or isinstance(in_field, itir.Literal):
            transformed_inputs.append(in_field)
        else:
            raise ValueError(f"Unsupported type {type(in_field)}")

    transformed_call = im.as_fieldop(stencil, SymbolicDomain.as_expr(target_domain))(
        *transformed_inputs
    )

    accessed_domains_without_tmp = {
        k: v
        for k, v in accessed_domains.items()
        if not k.startswith(tmp_uid_gen.prefix)  # type: ignore[arg-type] # prefix is always str
    }

    return transformed_call, accessed_domains_without_tmp


def _validate_temporary_usage(body: list[itir.Stmt], temporaries: list[str]):
    assigned_targets = set()
    for stmt in body:
        assert isinstance(stmt, itir.SetAt)  # TODO: extend for if-statements when they land
        assert isinstance(
            stmt.target, itir.SymRef
        )  # TODO: stmt.target can be an expr, e.g. make_tuple
        if stmt.target.id in assigned_targets:
            raise ValueError("Temporaries can only be used once within a program.")
        if stmt.target.id in temporaries:
            assigned_targets.add(stmt.target.id)


def infer_program(
    program: itir.Program,
    offset_provider: Dict[str, Dimension],
) -> itir.Program:
    accessed_domains: dict[str, SymbolicDomain] = {}
    transformed_set_ats: list[itir.SetAt] = []

    temporaries: list[str] = [tmp.id for tmp in program.declarations]

    _validate_temporary_usage(program.body, temporaries)

    for set_at in reversed(program.body):
        assert isinstance(set_at, itir.SetAt)
        if isinstance(set_at.expr, itir.SymRef):
            transformed_set_ats.insert(0, set_at)
            continue
        assert isinstance(set_at.expr, itir.FunCall)
        assert cpm.is_call_to(set_at.expr.fun, "as_fieldop")
        assert isinstance(
            set_at.target, itir.SymRef
        )  # TODO: stmt.target can be an expr, e.g. make_tuple
        if set_at.target.id in temporaries:
            # ignore temporaries as their domain is the `AUTO_DOMAIN` placeholder
            assert set_at.domain == AUTO_DOMAIN
        else:
            accessed_domains[set_at.target.id] = SymbolicDomain.from_expr(set_at.domain)

        transformed_as_fieldop, current_accessed_domains = infer_as_fieldop(
            set_at.expr, accessed_domains[set_at.target.id], offset_provider
        )
        transformed_set_ats.insert(
            0,
            itir.SetAt(
                expr=transformed_as_fieldop,
                domain=SymbolicDomain.as_expr(accessed_domains[set_at.target.id]),
                target=set_at.target,
            ),
        )

        for field in current_accessed_domains:
            if field in accessed_domains:
                # multiple accesses to the same field -> compute union of accessed domains
                if field in temporaries:
                    accessed_domains[field] = domain_union(
                        [accessed_domains[field], current_accessed_domains[field]]
                    )
                else:
                    # TODO(tehrengruber): if domain_ref is an external field the domain must
                    #  already be larger. This should be checked, but would require additions
                    #  to the IR.
                    pass
            else:
                accessed_domains[field] = current_accessed_domains[field]

    new_declarations = program.declarations
    for temporary in new_declarations:
        temporary.domain = SymbolicDomain.as_expr(accessed_domains[temporary.id])

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=new_declarations,
        body=transformed_set_ats,
    )
