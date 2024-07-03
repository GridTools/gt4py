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
import dataclasses



from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, SymbolicRange, domain_union
from gt4py.eve.extended_typing import Dict, List, Tuple, Union


# Define a mapping for offset values to dimension names and kinds
OFFSET_TO_DIMENSION = {
    itir.SymbolRef('Ioff'): ('IDim', DimensionKind.HORIZONTAL),
    itir.SymbolRef('Joff'): ('JDim', DimensionKind.HORIZONTAL),
    itir.SymbolRef('Koff'): ('KDim', DimensionKind.VERTICAL)
}


@dataclasses.dataclass(frozen=True)
class InferDomain:

    @staticmethod
    def _infer_dimension_from_offset(offset: itir.OffsetLiteral) -> Dimension:
        if offset.value in OFFSET_TO_DIMENSION:
            name, kind = OFFSET_TO_DIMENSION[offset.value]
            return Dimension(name, kind)
        else:
            raise ValueError("offset must be either Ioff, Joff, or Koff")

    @staticmethod
    def _extract_axis_dims(domain_expr: SymbolicDomain | itir.FunCall) -> List[str]:
        axis_dims = []
        if isinstance(domain_expr, SymbolicDomain) and domain_expr.grid_type == 'cartesian_domain':
            axis_dims.extend(domain_expr.ranges.keys())
        elif isinstance(domain_expr, itir.FunCall) and domain_expr.fun == im.ref('cartesian_domain'):
            for named_range in domain_expr.args:
                if isinstance(named_range, itir.FunCall) and named_range.fun == im.ref('named_range'):
                    axis_literal = named_range.args[0]
                    if isinstance(axis_literal, itir.AxisLiteral):
                        axis_dims.append(axis_literal.value)
        return axis_dims

    @staticmethod
    def _get_symbolic_domain(domain: Union[SymbolicDomain, itir.FunCall]) -> SymbolicDomain:
        if isinstance(domain, SymbolicDomain):
            return domain
        if isinstance(domain, itir.FunCall) and domain.fun == im.ref('cartesian_domain'):
            return SymbolicDomain.from_expr(domain)
        raise TypeError("domain must either be a FunCall or a SymbolicDomain.")

    @classmethod
    def _translate_domain(cls, symbolic_domain: SymbolicDomain, shift: Tuple[itir.OffsetLiteral, int],
                          dims: List[str]) -> SymbolicDomain:
        new_ranges = {dim: symbolic_domain.ranges[dim] for dim in dims}
        if shift:
            off, val = shift
            current_dim = cls._infer_dimension_from_offset(off)
            new_ranges[current_dim.value] = SymbolicRange.translate(
                symbolic_domain.ranges[current_dim.value], val.value
            )
        return SymbolicDomain("cartesian_domain", new_ranges)

    @classmethod
    def infer_as_fieldop(
            cls,
            stencil: itir.FunCall,
            input_domain: SymbolicDomain
    ) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]: # todo: test scan operator

        assert isinstance(stencil, itir.FunCall) and isinstance(stencil.fun, itir.FunCall)
        assert stencil.fun.fun == im.ref('as_fieldop')

        inputs = stencil.args
        tmp_counter = 0
        inputs_node = inputs.copy()

        # Set inputs for StencilClosure node by replacing FunCalls with temporary SymRefs
        for idx, in_field in enumerate(inputs):
            if isinstance(in_field, itir.FunCall):
                in_field.id = im.ref(f'__dom_inf_{tmp_counter}')
                inputs_node[idx] = im.ref(in_field.id)
                tmp_counter += 1

        accessed_domains = {str(in_field.id): [] for in_field in inputs}
        out_field_name = "tmp"  # todo: can this be derived from somewhere?

        symbolic_domain = cls._get_symbolic_domain(input_domain)

        # Build node that can be passed to TraceShifts
        node = itir.StencilClosure(
            stencil=stencil.fun.args[0],
            inputs=inputs_node,
            output=im.ref(out_field_name),
            domain=SymbolicDomain.as_expr(symbolic_domain),
        )

        # Extract the shifts and translate the domains accordingly
        for in_field in inputs:
            in_field_id = str(in_field.id)
            shifts_list = TraceShifts.apply(node)[in_field_id]
            dims = cls._extract_axis_dims(SymbolicDomain.as_expr(symbolic_domain))

            new_domains = [
                cls._translate_domain(symbolic_domain, shift, dims)
                for shift in shifts_list
            ]

            accessed_domains[in_field_id] = domain_union(new_domains)

        inputs_new = inputs.copy()
        for idx, in_field in enumerate(inputs):
            # Recursively traverse inputs
            if isinstance(in_field, itir.FunCall):
                transformed_calls_tmp, accessed_domains_tmp = cls.infer_as_fieldop(in_field,
                                                                                   accessed_domains[str(in_field.id)])
                inputs_new[idx] = transformed_calls_tmp

                # Merge accessed_domains and accessed_domains_tmp
                for k, v in accessed_domains_tmp.items():
                    if k in accessed_domains:
                        accessed_domains[k] = domain_union([accessed_domains[k], v])
                    else:
                        accessed_domains[k] = v

        transformed_call = im.call(im.call('as_fieldop')(stencil.fun.args[0], SymbolicDomain.as_expr(symbolic_domain)))(
            *inputs_new)

        accessed_domains_without_tmp = {k: v for k, v in accessed_domains.items() if not k.startswith('__dom_inf_')}

        return transformed_call, accessed_domains_without_tmp
