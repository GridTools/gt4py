from eve import NodeTranslator
from functional.iterator import ir, type_inference
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.collect_shifts import CollectShifts
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.popup_tmps import PopupTmps
from functional.iterator.transforms.prune_closure_inputs import PruneClosureInputs


AUTO_DOMAIN = ir.SymRef(id="_gtmp_auto_domain")


def split_closures(node: ir.FencilDefinition):
    tmps: list[ir.SymRef] = []

    def handle_arg(arg):
        if isinstance(arg, ir.SymRef):
            return arg
        if (
            isinstance(arg, ir.FunCall)
            and isinstance(arg.fun, ir.FunCall)
            and isinstance(arg.fun.fun, ir.SymRef)
            and arg.fun.fun.id == "lift"
        ):
            assert len(arg.fun.args) == 1
            ref = ir.SymRef(id=f"_gtmp_{len(tmps)}")
            tmps.append(ir.Sym(id=ref.id))
            unlifted = ir.FunCall(fun=arg.fun.args[0], args=arg.args)
            todos.append((ref, unlifted))
            return ref
        raise AssertionError()

    closures = []
    for closure in reversed(node.closures):
        wrapped_stencil = ir.FunCall(fun=closure.stencil, args=closure.inputs)
        popped_stencil = PopupTmps().visit(wrapped_stencil)

        todos = [(closure.output, popped_stencil)]
        domain = closure.domain
        while todos:
            output, call = todos.pop()
            closure = ir.StencilClosure(
                domain=domain,
                stencil=call.fun,
                output=output,
                inputs=[handle_arg(arg) for arg in call.args],
            )
            closures.append(closure)
            domain = AUTO_DOMAIN

    return ir.FencilDefinition(
        id=node.id, params=node.params + tmps, closures=list(reversed(closures))
    ), [tmp.id for tmp in tmps]


def update_cartesian_domains(node: ir.FencilDefinition, offset_provider):
    def extend(domain, shifts):
        assert isinstance(domain, ir.FunCall) and domain.fun == ir.SymRef(id="domain")
        assert all(isinstance(axis, CartesianAxis) for axis in offset_provider.values())

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
                            args=[lower_bound, ir.IntLiteral(value=lower_offset)],
                        )
                        if lower_offset
                        else lower_bound,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[upper_bound, ir.IntLiteral(value=upper_offset)],
                        )
                        if upper_offset
                        else upper_bound,
                    ],
                )
            )

        return ir.FunCall(fun=domain.fun, args=named_ranges)

    closures = []
    shifts: dict[str, list[tuple]] = dict()
    domain = None
    for closure in reversed(node.closures):
        if closure.domain == AUTO_DOMAIN:
            output_shifts = shifts.get(closure.output.id, [])
            domain = extend(domain, output_shifts)
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )
        else:
            domain = closure.domain
            shifts = dict()

        closures.append(closure)

        if closure.stencil == ir.SymRef(id="deref"):
            continue

        local_shifts: dict[str, list[tuple]] = dict()
        if isinstance(closure.stencil, ir.FunCall) and closure.stencil.fun == ir.SymRef(id="scan"):
            # get params of scan function, but ignore accumulator
            fun = closure.stencil.args[0]
            params = fun.params[1:]
        else:
            assert isinstance(closure.stencil, ir.Lambda)
            fun = closure.stencil
            params = fun.params
        CollectShifts().visit(fun, shifts=local_shifts)
        input_map = {param.id: inp.id for param, inp in zip(params, closure.inputs)}
        for param, shift in local_shifts.items():
            shifts.setdefault(input_map[param], []).extend(shift)

    return ir.FencilDefinition(id=node.id, params=node.params, closures=list(reversed(closures)))


def collect_tmps_info(node: ir.FencilDefinition, tmps):
    domains = {
        closure.output.id: closure.domain for closure in node.closures if closure.output.id in tmps
    }

    def convert_type(dtype):
        if isinstance(dtype, type_inference.Primitive):
            return dtype.name
        if isinstance(dtype, type_inference.Var):
            return dtype.idx
        if isinstance(dtype, type_inference.PartialTupleVar):
            elems_dict = dict(dtype.elems)
            assert len(elems_dict) == max(elems_dict) + 1
            return tuple(convert_type(elems_dict[i]) for i in range(len(elems_dict)))
        assert isinstance(dtype, type_inference.Tuple)
        return tuple(convert_type(e) for e in dtype.elems)

    fencil_type = type_inference.infer(node)
    assert isinstance(fencil_type, type_inference.Fencil)
    assert isinstance(fencil_type.params, type_inference.Tuple)
    all_types = []
    types = dict()
    for param, dtype in zip(node.params, fencil_type.params.elems):
        assert isinstance(dtype, type_inference.Val)
        all_types.append(convert_type(dtype.dtype))
        if param.id in tmps:
            assert param.id not in types
            t = all_types[-1]
            types[param.id] = all_types.index(t) if isinstance(t, int) else t

    return {tmp: (domains[tmp], types[tmp]) for tmp in tmps}


class CreateGlobalTmps(NodeTranslator):
    def visit_FencilDefinition(self, node: ir.FencilDefinition, *, offset_provider, register_tmp):
        node, tmps = split_closures(node)
        node = PruneClosureInputs().visit(node)
        node = EtaReduction().visit(node)
        node = update_cartesian_domains(node, offset_provider)
        if register_tmp is not None:
            infos = collect_tmps_info(node, tmps)
            for tmp, (domain, dtype) in infos.items():
                register_tmp(node.id, tmp, domain, dtype)
        return node
