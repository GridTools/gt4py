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
    domain_union, AUTO_DOMAIN,
)
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm

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


# TODO: move to SymbolicDomain.translate?
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
def trace_shifts(stencil: itir.Expr, input_ids: list[str], domain: itir.Expr):
    node = itir.StencilClosure(
        stencil=stencil,
        inputs=[im.ref(id_) for id_ in input_ids],
        output=im.ref("__dummy"),
        domain=domain,
    )
    return TraceShifts.apply(node)


def infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:  # todo: test scan operator
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")

    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    input_ids: list[str] = []
    accessed_domains: Dict[str, SymbolicDomain] = {}

    # Assign ids for all inputs to `as_fieldop`. `SymRef`s stay as is, nested `as_fieldop` get a
    # temporary id.
    tmp_uid_gen = eve_utils.UIDGenerator(prefix="__dom_inf")
    for in_field in inputs:
        if isinstance(in_field, itir.FunCall):
            id_ = tmp_uid_gen.sequential_id()
        else:
            assert isinstance(in_field, itir.SymRef)
            id_ = in_field.id
        input_ids.append(id_)
        accessed_domains[id_] = []

    if isinstance(input_domain, itir.FunCall):
        input_domain = SymbolicDomain.from_expr(input_domain)

    # Extract the shifts and translate the domains accordingly
    shifts_results = trace_shifts(
        stencil, input_ids, SymbolicDomain.as_expr(input_domain)
    )

    for in_field_id in input_ids:
        shifts_list = shifts_results[in_field_id]

        new_domains = [
            _translate_domain(input_domain, shift, offset_provider) for shift in shifts_list
        ]

        accessed_domains[in_field_id] = domain_union(new_domains)

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldops`
    transformed_inputs = []
    for in_field_id, in_field in zip(input_ids, inputs):
        if isinstance(in_field, itir.FunCall):
            transformed_input, accessed_domains_tmp = infer_as_fieldop(
                in_field, accessed_domains[in_field_id], offset_provider
            )
            transformed_inputs.append(transformed_input)

            # Merge accessed_domains and accessed_domains_tmp
            accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)
        else:
            assert isinstance(in_field, itir.SymRef)
            transformed_inputs.append(in_field)

    transformed_call = im.as_fieldop(stencil, SymbolicDomain.as_expr(input_domain))(*transformed_inputs)

    accessed_domains_without_tmp = {
        k: v for k, v in accessed_domains.items() if not k.startswith(tmp_uid_gen.prefix)
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
    # TODO: stmt.target can be an expr, e.g. make_tuple
    tmp_assignments = [stmt.target.id for stmt in body if stmt.target.id in temporaries]
    if len(tmp_assignments) != len(set(tmp_assignments)):
        raise ValueError("Temporaries can only be used once within a program.")


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
        assert isinstance(set_at.expr, itir.FunCall)
        assert cpm.is_call_to(set_at.expr.fun, "as_fieldop")

        if set_at.target.id in temporaries:
            # ignore temporaries as their domain is the `AUTO_DOMAIN` placeholder
            assert set_at.domain == AUTO_DOMAIN
        else:
            accessed_domains[set_at.target.id] = SymbolicDomain.from_expr(set_at.domain)

        actual_call, current_accessed_domains = infer_as_fieldop(
            set_at.expr, accessed_domains[set_at.target.id], offset_provider
        )
        transformed_set_ats.insert(
            0,
            itir.SetAt(expr=actual_call, domain=actual_call.fun.args[1], target=set_at.target),
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
