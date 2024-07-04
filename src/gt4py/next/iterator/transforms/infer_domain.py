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

from gt4py.eve.extended_typing import Dict, List, Tuple, Union
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, SymbolicRange, domain_union
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts


# Define a mapping for offset values to dimension names and kinds
OFFSET_TO_DIMENSION = {
    itir.SymbolRef("Ioff"): ("IDim", DimensionKind.HORIZONTAL),
    itir.SymbolRef("Joff"): ("JDim", DimensionKind.HORIZONTAL),
    itir.SymbolRef("Koff"): ("KDim", DimensionKind.VERTICAL),
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
        if isinstance(domain_expr, SymbolicDomain) and domain_expr.grid_type == "cartesian_domain":
            axis_dims.extend(domain_expr.ranges.keys())
        elif isinstance(domain_expr, itir.FunCall) and domain_expr.fun == im.ref(
            "cartesian_domain"
        ):
            for named_range in domain_expr.args:
                if isinstance(named_range, itir.FunCall) and named_range.fun == im.ref(
                    "named_range"
                ):
                    axis_literal = named_range.args[0]
                    if isinstance(axis_literal, itir.AxisLiteral):
                        axis_dims.append(axis_literal.value)
        return axis_dims

    @staticmethod
    def _get_symbolic_domain(domain: Union[SymbolicDomain, itir.FunCall]) -> SymbolicDomain:
        if isinstance(domain, SymbolicDomain):
            return domain
        if isinstance(domain, itir.FunCall) and domain.fun == im.ref("cartesian_domain"):
            return SymbolicDomain.from_expr(domain)
        raise TypeError("domain must either be a FunCall or a SymbolicDomain.")

    @staticmethod
    def _merge_domains(original_domains, new_domains):
        for key, value in new_domains.items():
            if key in original_domains:
                original_domains[key] = domain_union([original_domains[key], value])
            else:
                original_domains[key] = value

        return {
            key: domain_union(value) if isinstance(value, list) else value
            for key, value in original_domains.items()
        }

    @classmethod
    def _translate_domain(
        cls, symbolic_domain: SymbolicDomain, shift: Tuple[itir.OffsetLiteral, int], dims: List[str]
    ) -> SymbolicDomain:
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
        cls, applied_fieldop: itir.FunCall, input_domain: SymbolicDomain
    ) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:  # todo: test scan operator
        assert isinstance(applied_fieldop, itir.FunCall) and isinstance(
            applied_fieldop.fun, itir.FunCall
        )
        assert applied_fieldop.fun.fun == im.ref("as_fieldop")

        stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

        inputs_node = []
        accessed_domains = {}

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

        symbolic_domain = cls._get_symbolic_domain(input_domain)

        # TODO: until TraceShifts directly supporty stencils we just wrap our expression into a dummy closure in this helper function.
        def trace_shifts(stencil: itir.Expr, inputs: list[itir.Expr], domain: itir.Expr):
            node = itir.StencilClosure(
                stencil=stencil,
                inputs=inputs,
                output=im.ref(out_field_name),
                domain=domain,
            )
            return TraceShifts.apply(node)

        # Extract the shifts and translate the domains accordingly
        shifts_results = trace_shifts(stencil, inputs_node, SymbolicDomain.as_expr(symbolic_domain))
        dims = cls._extract_axis_dims(SymbolicDomain.as_expr(symbolic_domain))

        for in_field in inputs:
            in_field_id = str(in_field.id)
            shifts_list = shifts_results[in_field_id]

            new_domains = [
                cls._translate_domain(symbolic_domain, shift, dims) for shift in shifts_list
            ]

            accessed_domains[in_field_id] = domain_union(new_domains)

        inputs_new = []
        for in_field in inputs:
            # Recursively traverse inputs
            if isinstance(in_field, itir.FunCall):
                transformed_calls_tmp, accessed_domains_tmp = cls.infer_as_fieldop(
                    in_field, accessed_domains[str(in_field.id)]
                )
                inputs_new.append(transformed_calls_tmp)

                # Merge accessed_domains and accessed_domains_tmp
                accessed_domains = cls._merge_domains(accessed_domains, accessed_domains_tmp)
            else:
                inputs_new.append(in_field)

        transformed_call = im.call(
            im.call("as_fieldop")(stencil, SymbolicDomain.as_expr(symbolic_domain))
        )(*inputs_new)

        accessed_domains_without_tmp = {
            k: v for k, v in accessed_domains.items() if not k.startswith("__dom_inf_")
        }

        return transformed_call, accessed_domains_without_tmp
