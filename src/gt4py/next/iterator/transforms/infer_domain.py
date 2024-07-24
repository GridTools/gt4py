# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.eve.extended_typing import Dict, Tuple
from gt4py.next import common
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import (
    SymbolicDomain,
    SymbolicRange,
    _max_domain_sizes_by_location_type,
    domain_union,
)
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts


def _merge_domains(
    original_domains: Dict[str, SymbolicDomain], new_domains: Dict[str, SymbolicDomain]
) -> Dict[str, SymbolicDomain]:
    for key, value in new_domains.items():
        if key in original_domains:
            original_domains[key] = domain_union([original_domains[key], value])
        else:
            original_domains[key] = value

    return {
        key: domain_union(value) if isinstance(value, list) else value
        for key, value in original_domains.items()
    }


def _translate_domain(
    symbolic_domain: SymbolicDomain,
    shift: Tuple[itir.OffsetLiteral, itir.OffsetLiteral],
    offset_provider: Dict[str, Dimension],
) -> SymbolicDomain:
    dims = list(symbolic_domain.ranges.keys())
    new_ranges = {dim: symbolic_domain.ranges[dim] for dim in dims}
    if shift:
        off, val = shift
        nbt_provider = offset_provider[off.value]
        if isinstance(nbt_provider, common.Dimension):
            current_dim = nbt_provider
            # cartesian offset
            new_ranges[current_dim] = SymbolicRange.translate(
                symbolic_domain.ranges[current_dim], val.value
            )
        elif isinstance(nbt_provider, common.Connectivity):
            # unstructured shift
            # TODO: move to initialization
            horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)

            old_dim = nbt_provider.origin_axis
            new_dim = nbt_provider.neighbor_axis

            assert new_dim not in new_ranges or old_dim == new_dim

            # TODO(tehrengruber): symbolic sizes for ICON?
            new_range = SymbolicRange(
                im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                im.literal(str(horizontal_sizes[new_dim.value]), itir.INTEGER_INDEX_BUILTIN),
            )
            new_ranges = dict(
                (dim, range_) if dim != old_dim else (new_dim, new_range)
                for dim, range_ in new_ranges.items()
            )
        else:
            raise AssertionError()

    return SymbolicDomain(symbolic_domain.grid_type, new_ranges)


# TODO: until TraceShifts directly supporty stencils we just wrap our expression into a dummy closure in this helper function.
def trace_shifts(stencil: itir.Expr, inputs: list[itir.Expr], domain: itir.Expr, out_field_name):
    node = itir.StencilClosure(
        stencil=stencil,
        inputs=inputs,
        output=im.ref(out_field_name),
        domain=domain,
    )
    return TraceShifts.apply(node)


def infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:  # todo: test scan operator
    assert isinstance(applied_fieldop, itir.FunCall) and isinstance(
        applied_fieldop.fun, itir.FunCall
    )
    assert applied_fieldop.fun.fun == im.ref("as_fieldop")

    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    inputs_node = []
    accessed_domains: Dict[str, SymbolicDomain] = {}

    # Set inputs for StencilClosure node by replacing FunCalls with temporary SymRefs
    tmp_counter = 0
    for in_field in inputs:
        if isinstance(in_field, itir.FunCall):
            in_field.id = im.ref(f"__dom_inf_{tmp_counter}")
            inputs_node.append(im.ref(in_field.id))
            accessed_domains[str(in_field.id)] = []
            tmp_counter += 1
        else:
            inputs_node.append(in_field)
            accessed_domains[str(in_field.id)] = []

    out_field_name = "tmp"  # todo: can this be derived from somewhere?

    if isinstance(input_domain, itir.FunCall):
        input_domain = SymbolicDomain.from_expr(input_domain)

    # Extract the shifts and translate the domains accordingly
    shifts_results = trace_shifts(
        stencil, inputs_node, SymbolicDomain.as_expr(input_domain), out_field_name
    )

    for in_field in inputs:
        in_field_id = str(in_field.id)
        shifts_list = shifts_results[in_field_id]

        new_domains = [
            _translate_domain(input_domain, shift, offset_provider) for shift in shifts_list
        ]

        accessed_domains[in_field_id] = domain_union(new_domains)

    inputs_new = []
    for in_field in inputs:
        # Recursively traverse inputs
        if isinstance(in_field, itir.FunCall):
            transformed_calls_tmp, accessed_domains_tmp = infer_as_fieldop(
                in_field, accessed_domains[str(in_field.id)], offset_provider
            )
            inputs_new.append(transformed_calls_tmp)

            # Merge accessed_domains and accessed_domains_tmp
            accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)
        else:
            inputs_new.append(in_field)

    transformed_call = im.call(
        im.call("as_fieldop")(stencil, SymbolicDomain.as_expr(input_domain))
    )(*inputs_new)

    accessed_domains_without_tmp = {
        k: v for k, v in accessed_domains.items() if not k.startswith("__dom_inf_")
    }

    return transformed_call, accessed_domains_without_tmp

def infer_let( # Todo generaize for nested lets
    applied_let: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    assert isinstance(applied_let, itir.FunCall) and isinstance(
        applied_let.fun, itir.Lambda
    )
    if applied_let.fun.expr.fun.fun == im.ref("as_fieldop"):
        transformed_calls_expr, accessed_domains_expr = infer_as_fieldop(
            applied_let.fun.expr, input_domain, offset_provider
        )
    if applied_let.args[0].fun.fun == im.ref("as_fieldop"):
        transformed_calls_args, accessed_domains_args= infer_as_fieldop(
            applied_let.args[0], accessed_domains_expr[applied_let.fun.params[0].id], offset_provider # Todo generaize for more inputs
        )
    accessed_domains = accessed_domains_args
    transformed_call = im.let(applied_let.fun.params[0].id,transformed_calls_args)(transformed_calls_expr)
    return transformed_call, accessed_domains


def _validate_temporary_usage(body: list[itir.SetAt], temporaries: list[str]):
    for i in range(len(body)):
        for j in range(i + 1, len(body)):
            if (
                str(body[i].target.id) == str(body[j].target.id)
                and str(body[i].target.id) in temporaries
            ):
                raise ValueError("Temporaries can only be used once within a program.")


def infer_program(
    program: itir.Program,
    offset_provider: Dict[str, Dimension],
) -> itir.Program:
    fields_dict = {}
    new_set_at_list = []

    temporaries = [str(tmp.id) for tmp in program.declarations]

    _validate_temporary_usage(program.body, temporaries)

    for set_at in reversed(program.body):
        assert isinstance(set_at, itir.SetAt)
        assert isinstance(set_at.expr, itir.FunCall)
        assert isinstance(set_at.expr.fun, itir.FunCall)
        assert set_at.expr.fun.fun == im.ref("as_fieldop")

        if str(set_at.target.id) not in temporaries:
            fields_dict[set_at.target] = SymbolicDomain.from_expr(set_at.domain)

        actual_call, actual_domains = infer_as_fieldop(
            set_at.expr, fields_dict[set_at.target], offset_provider
        )
        new_set_at_list.insert(
            0,
            itir.SetAt(expr=actual_call, domain=actual_call.fun.args[1], target=set_at.target),
        )

        for domain in actual_domains:
            domain_ref = itir.SymRef(id=domain)
            if domain_ref in fields_dict:
                fields_dict[domain_ref] = domain_union(
                    [fields_dict[domain_ref], actual_domains[domain]]
                )
            else:
                fields_dict[domain_ref] = actual_domains[domain]

    new_declarations = program.declarations
    for temporary in new_declarations:
        temporary.domain = SymbolicDomain.as_expr(fields_dict[itir.SymRef(id=temporary.id)])

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=new_declarations,
        body=new_set_at_list,
    )
