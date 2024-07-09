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

from gt4py.eve.extended_typing import Dict, Tuple, Union
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, SymbolicRange, domain_union
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts


@dataclasses.dataclass(frozen=True)
class InferDomain:
    @staticmethod
    def _get_symbolic_domain(domain: Union[SymbolicDomain, itir.FunCall]) -> SymbolicDomain:
        if isinstance(domain, SymbolicDomain):
            return domain
        if isinstance(domain, itir.FunCall) and domain.fun == im.ref("cartesian_domain"):
            return SymbolicDomain.from_expr(domain)
        raise TypeError("domain must either be a FunCall or a SymbolicDomain.")

    @staticmethod
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

    @classmethod
    def _translate_domain(
        cls,
        symbolic_domain: SymbolicDomain,
        shift: Tuple[itir.OffsetLiteral, itir.OffsetLiteral],
        offset_provider: Dict[str, Dimension],
    ) -> SymbolicDomain:
        dims = list(symbolic_domain.ranges.keys())
        new_ranges = {dim: symbolic_domain.ranges[dim] for dim in dims}
        if shift:
            off, val = shift
            current_dim = offset_provider[off.value]
            new_ranges[current_dim] = SymbolicRange.translate(
                symbolic_domain.ranges[current_dim], val.value
            )
        return SymbolicDomain("cartesian_domain", new_ranges)

    @classmethod
    def infer_as_fieldop(
        cls,
        applied_fieldop: itir.FunCall,
        input_domain: SymbolicDomain,
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

        for in_field in inputs:
            in_field_id = str(in_field.id)
            shifts_list = shifts_results[in_field_id]

            new_domains = [
                cls._translate_domain(symbolic_domain, shift, offset_provider)
                for shift in shifts_list
            ]

            accessed_domains[in_field_id] = domain_union(new_domains)

        inputs_new = []
        for in_field in inputs:
            # Recursively traverse inputs
            if isinstance(in_field, itir.FunCall):
                transformed_calls_tmp, accessed_domains_tmp = cls.infer_as_fieldop(
                    in_field, accessed_domains[str(in_field.id)], offset_provider
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

    @staticmethod
    def _validate_temporary_usage(body: list[itir.SetAt], temporaries: list[str]):
        for i in range(len(body)):
            for j in range(i + 1, len(body)):
                if (
                    str(body[i].target.id) == str(body[j].target.id)
                    and str(body[i].target.id) in temporaries
                ):
                    raise ValueError("Temporaries can only be used once within a program.")

    @classmethod
    def infer_program(
        cls,
        program: itir.Program,
        offset_provider: Dict[str, Dimension],
    ) -> itir.Program:
        fields_dict = {}
        new_set_at_list = []

        temporaries = [str(tmp.id) for tmp in program.declarations]

        cls._validate_temporary_usage(program.body, temporaries)

        for set_at in reversed(program.body):
            assert isinstance(set_at, itir.SetAt)
            assert isinstance(set_at.expr, itir.FunCall)
            assert isinstance(set_at.expr.fun, itir.FunCall)
            assert set_at.expr.fun.fun == im.ref("as_fieldop")

            if str(set_at.target.id) not in temporaries:
                fields_dict[set_at.target] = SymbolicDomain.from_expr(set_at.domain)

            actual_call, actual_domains = InferDomain.infer_as_fieldop(
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
