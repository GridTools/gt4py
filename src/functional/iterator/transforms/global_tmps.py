from typing import Dict, List

from eve import NodeTranslator
from functional.iterator import ir
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.collect_shifts import CollectShifts
from functional.iterator.transforms.popup_tmps import PopupTmps


class CreateGlobalTmps(NodeTranslator):
    @staticmethod
    def _extend_domain(domain: ir.FunCall, offset_provider, shifts):
        assert isinstance(domain.fun, ir.SymRef) and domain.fun.id == "domain"
        assert all(isinstance(o, CartesianAxis) for o in offset_provider.values())

        offset_limits = {k: (0, 0) for k in offset_provider.keys()}

        for shift in shifts:
            offsets = {k: 0 for k in offset_provider.keys()}
            for k, v in zip(shift[0::2], shift[1::2]):
                offsets[k.value] += v.value
            for k, v in offsets.items():
                old_min, old_max = offset_limits[k]
                offset_limits[k] = (min(old_min, v), max(old_max, v))

        offset_limits = {v.value: offset_limits[k] for k, v in offset_provider.items()}

        named_ranges = []
        for named_range in domain.args:
            assert (
                isinstance(named_range, ir.FunCall)
                and isinstance(named_range.fun, ir.SymRef)
                and named_range.fun.id == "named_range"
            )
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, ir.AxisLiteral)

            lower_offset, upper_offset = offset_limits.get(axis_literal.value, (0, 0))
            named_ranges.append(
                ir.FunCall(
                    fun=named_range.fun,
                    args=[
                        axis_literal,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[
                                lower_bound,
                                ir.Literal(value=str(lower_offset), type="int"),
                            ],
                        )
                        if lower_offset
                        else lower_bound,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[
                                upper_bound,
                                ir.Literal(value=str(upper_offset), type="int"),
                            ],
                        )
                        if upper_offset
                        else upper_bound,
                    ],
                )
            )

        return ir.FunCall(fun=domain.fun, args=named_ranges)

    def visit_FencilDefinition(self, node: ir.FencilDefinition, *, offset_provider, register_tmp):
        tmps: List[ir.Sym] = []

        def handle_arg(arg):
            if isinstance(arg, ir.SymRef):
                return arg
            if (
                isinstance(arg, ir.FunCall)
                and isinstance(arg.fun, ir.FunCall)
                and arg.fun.fun.id == "lift"
            ):
                ref = ir.SymRef(id=f"tmp{len(tmps)}")
                tmps.append(ir.Sym(id=ref.id))
                assert len(arg.fun.args) == 1
                unlifted = ir.FunCall(fun=arg.fun.args[0], args=arg.args)
                todos.append((ref, unlifted))
                return ref
            raise AssertionError()

        closures = []
        tmp_domains = dict()
        for closure in reversed(node.closures):
            if not isinstance(closure.stencil, ir.Lambda):
                if not isinstance(closure.stencil, ir.SymRef) or closure.stencil.id != "deref":
                    raise NotImplementedError()
                closure = ir.StencilClosure(
                    domain=closure.domain,
                    stencil=ir.Lambda(
                        params=[ir.Sym(id="x")],
                        expr=ir.FunCall(fun=closure.stencil, args=[ir.SymRef(id="x")]),
                    ),
                    output=closure.output,
                    inputs=closure.inputs,
                )
            wrapped_stencil = ir.FunCall(fun=closure.stencil, args=closure.inputs)
            popped_stencil = PopupTmps().visit(wrapped_stencil)
            todos = [(closure.output, popped_stencil)]

            shifts: Dict[str, List[tuple]] = dict()
            domain = closure.domain
            while todos:
                output, call = todos.pop()
                output_shifts: List[tuple] = shifts.get(output.id, [])
                domain = self._extend_domain(domain, offset_provider, output_shifts)
                if output.id in {tmp.id for tmp in tmps}:
                    assert output.id not in tmp_domains
                    tmp_domains[output.id] = domain
                closure = ir.StencilClosure(
                    domain=domain,
                    stencil=call.fun,
                    output=output,
                    inputs=[handle_arg(arg) for arg in call.args],
                )
                local_shifts: Dict[str, List[tuple]] = dict()
                CollectShifts().visit(closure.stencil, shifts=local_shifts)
                input_map = {
                    param.id: arg.id for param, arg in zip(closure.stencil.params, closure.inputs)
                }
                for k, v in local_shifts.items():
                    shifts.setdefault(input_map[k], []).extend(v)
                closures.append(closure)

        assert {tmp.id for tmp in tmps} == set(tmp_domains.keys())
        if register_tmp is not None:
            for tmp, domain in tmp_domains.items():
                register_tmp(tmp, domain)

        return ir.FencilDefinition(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params + tmps,
            closures=list(reversed(closures)),
        )
