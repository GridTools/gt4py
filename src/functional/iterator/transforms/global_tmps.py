from typing import Any, Optional

from eve import Coerced, NodeTranslator
from eve.traits import SymbolTableTrait
from functional.iterator import ir, type_inference
from functional.iterator.pretty_printer import PrettyPrinter
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.collect_shifts import CollectShifts
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.popup_tmps import PopupTmps
from functional.iterator.transforms.prune_closure_inputs import PruneClosureInputs


"""Iterator IR extension for global temporaries.

Replaces lifted function calls by temporaries using the following steps:
1. Split closures by popping up lifted function calls to the top of the expression tree, (that is,
   to stencil arguments) and then extracting them as new closures.
2. Introduces a new fencil-scope variable (the temporary) for each output of newly created closures.
   The domain size is set to a new symbol `_gtmp_auto_domain`.
3. Infer the domain sizes for the new closures by analysing the accesses/shifts within all closures
   and replace all occurrences of `_gtmp_auto_domain` by concrete domain sizes.
4. Infer the data type and size of the temporary buffers.
"""


AUTO_DOMAIN = ir.SymRef(id="_gtmp_auto_domain")


# Iterator IR extension nodes


class Temporary(ir.Node):
    """Iterator IR extension: declaration of a temporary buffer."""

    id: Coerced[ir.SymbolName]  # noqa: A003
    domain: Optional[ir.Expr] = None
    dtype: Optional[Any] = None


class FencilWithTemporaries(ir.Node, SymbolTableTrait):
    """Iterator IR extension: declaration of a fencil with temporary buffers."""

    fencil: ir.FencilDefinition
    params: list[ir.Sym]
    tmps: list[Temporary]


# Extensions for `PrettyPrinter` for easier debugging


def pformat_Temporary(printer: PrettyPrinter, node: Temporary, *, prec: int) -> list[str]:
    start, end = [node.id + " = temporary("], [");"]
    args = []
    if node.domain is not None:
        args.append(printer._hmerge(["domain="], printer.visit(node.domain, prec=0)))
    if node.dtype is not None:
        args.append(printer._hmerge(["dtype="], [str(node.dtype)]))
    hargs = printer._hmerge(*printer._hinterleave(args, ", "))
    vargs = printer._vmerge(*printer._hinterleave(args, ","))
    oargs = printer._optimum(hargs, vargs)
    h = printer._hmerge(start, oargs, end)
    v = printer._vmerge(start, printer._indent(oargs), end)
    return printer._optimum(h, v)


def pformat_FencilWithTemporaries(
    printer: PrettyPrinter, node: FencilWithTemporaries, *, prec: int
) -> list[str]:
    assert prec == 0
    params = printer.visit(node.params, prec=0)
    fencil = printer.visit(node.fencil, prec=0)
    tmps = printer.visit(node.tmps, prec=0)
    args = params + [[tmp.id] for tmp in node.tmps]

    hparams = printer._hmerge([node.fencil.id + "("], *printer._hinterleave(params, ", "), [") {"])
    vparams = printer._vmerge(
        [node.fencil.id + "("], *printer._hinterleave(params, ",", indent=True), [") {"]
    )
    params = printer._optimum(hparams, vparams)

    hargs = printer._hmerge(*printer._hinterleave(args, ", "))
    vargs = printer._vmerge(*printer._hinterleave(args, ","))
    args = printer._optimum(hargs, vargs)

    fencil = printer._hmerge(fencil, [";"])

    hcall = printer._hmerge([node.fencil.id + "("], args, [");"])
    vcall = printer._vmerge(printer._hmerge([node.fencil.id + "("]), printer._indent(args), [");"])
    call = printer._optimum(hcall, vcall)

    body = printer._vmerge(*tmps, fencil, call)
    return printer._vmerge(params, printer._indent(body), ["}"])


PrettyPrinter.visit_Temporary = pformat_Temporary  # type: ignore
PrettyPrinter.visit_FencilWithTemporaries = pformat_FencilWithTemporaries  # type: ignore


# Main implementation


def split_closures(node: ir.FencilDefinition) -> FencilWithTemporaries:
    """Split closures on lifted function calls and introduce new temporary buffers for return values.

    Newly introduced temporaries will have the symbolic size of `AUTO_DOMAIN`. A symbol with the
    same name is also added as a fencil argument (to be replaced at a later stage).

    For each closure, follows these steps:
    1. Pops up lifted function calls to the top of the expression tree.
    2. Introduce new temporary for the output.
    3. Extract lifted function class as new closures with the previously created temporary as output.
    The closures are processed in reverse order to properly respect the dependencies.
    """
    tmps: list[ir.SymRef] = []

    def handle_arg(arg):
        """Handle arguments of closure calls: extract lifted function calls.

        Lifted function calls, do:
        1. Replace the call by a new symbol ref, put this into `tmps`.
        2. Put the ‘unlifted’ function call to the stack of stencil calls that still have to be
        processed.
        """
        if isinstance(arg, ir.SymRef):
            return arg
        if (
            isinstance(arg, ir.FunCall)
            and isinstance(arg.fun, ir.FunCall)
            and arg.fun.fun == ir.SymRef(id="lift")
        ):
            assert len(arg.fun.args) == 1
            ref = ir.SymRef(id=f"_gtmp_{len(tmps)}")
            tmps.append(ir.Sym(id=ref.id))
            unlifted = ir.FunCall(fun=arg.fun.args[0], args=arg.args)
            stencil_stack.append((ref, unlifted))
            return ref
        raise AssertionError()

    closures = []
    for closure in reversed(node.closures):
        wrapped_stencil = ir.FunCall(fun=closure.stencil, args=closure.inputs)
        popped_stencil = PopupTmps().visit(wrapped_stencil)

        stencil_stack = [(closure.output, popped_stencil)]
        domain = closure.domain
        while stencil_stack:
            output, call = stencil_stack.pop()
            closure = ir.StencilClosure(
                domain=domain,
                stencil=call.fun,
                output=output,
                inputs=[handle_arg(arg) for arg in call.args],
            )
            closures.append(closure)
            domain = AUTO_DOMAIN

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params + [ir.Sym(id=tmp.id) for tmp in tmps] + [ir.Sym(id=AUTO_DOMAIN.id)],
            closures=list(reversed(closures)),
        ),
        params=node.params,
        tmps=[Temporary(id=tmp.id) for tmp in tmps],
    )


def _collect_stencil_shifts(stencil, return_params=False):
    """Collect shift offsets applied within a stencil, correctly handling scans."""
    if isinstance(stencil, ir.FunCall) and stencil.fun == ir.SymRef(id="scan"):
        # get params of scan function, but ignore accumulator
        fun = stencil.args[0]
        params = fun.params[1:]
    else:
        assert isinstance(stencil, ir.Lambda)
        fun = stencil
        params = fun.params
    shifts: dict[str, list[tuple]] = dict()
    CollectShifts().visit(fun, shifts=shifts)
    if return_params:
        return shifts, params
    return shifts


def update_cartesian_domains(node: FencilWithTemporaries, offset_provider) -> FencilWithTemporaries:
    """Replace appearances of `AUTO_DOMAIN` by concrete domain sizes.

    Naive extent analysis, does not handle boundary conditions etc. in a smart way.
    """

    def extend(domain, shifts):
        if not any(shifts):
            return domain
        assert isinstance(domain, ir.FunCall) and domain.fun == ir.SymRef(id="cartesian_domain")
        assert all(isinstance(axis, CartesianAxis) for axis in offset_provider.values())

        offset_limits = {k: (0, 0) for k in offset_provider.keys()}
        for shift in shifts:
            offsets = {k: 0 for k in offset_provider.keys()}
            for k, v in zip(shift[0::2], shift[1::2]):
                assert isinstance(v, ir.OffsetLiteral) and isinstance(v.value, int)
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
                            args=[lower_bound, ir.Literal(value=str(lower_offset), type="int")],
                        )
                        if lower_offset
                        else lower_bound,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[upper_bound, ir.Literal(value=str(upper_offset), type="int")],
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
    for closure in reversed(node.fencil.closures):
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

        local_shifts, params = _collect_stencil_shifts(closure.stencil, return_params=True)
        input_map = {param.id: inp.id for param, inp in zip(params, closure.inputs)}
        for param, shift in local_shifts.items():
            shifts.setdefault(input_map[param], []).extend(shift)

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.fencil.params[:-1],
            closures=list(reversed(closures)),
        ),
        params=node.params,
        tmps=node.tmps,
    )


def update_unstructured_domains(node: FencilWithTemporaries, offset_provider):
    """Replace appearances of `AUTO_DOMAIN` by concrete domain sizes."""
    known_domains = {
        closure.output.id: closure.domain
        for closure in node.fencil.closures
        if closure.domain != AUTO_DOMAIN
    }

    closures = []
    for closure in node.fencil.closures:
        if closure.domain == AUTO_DOMAIN:
            if closure.stencil == ir.SymRef(id="deref"):
                domain = known_domains[closure.inputs[0].id]
            else:
                shifts = _collect_stencil_shifts(closure.stencil)
                first_connectivity = {c[0].value for s in shifts.values() for c in s if c}
                if not first_connectivity:
                    # for now we assume that all inputs have the same domain
                    assert all(
                        known_domains[inp.id] == known_domains[closure.inputs[0].id]
                        for inp in closure.inputs
                    )
                    domain = known_domains[closure.inputs[0].id]
                else:
                    assert len(first_connectivity) == 1
                    conn = offset_provider[next(iter(first_connectivity))]
                    axis = conn.origin_axis
                    size = conn.tbl.shape[0]
                    domain = ir.FunCall(
                        fun=ir.SymRef(id="unstructured_domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=axis.value),
                                    ir.Literal(value="0", type="int"),
                                    ir.Literal(value=str(size), type="int"),
                                ],
                            )
                        ],
                    )

            known_domains[closure.output.id] = domain
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )

        closures.append(closure)
    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.fencil.params[:-1],
            closures=closures,
        ),
        params=node.params,
        tmps=node.tmps,
    )


def collect_tmps_info(node: FencilWithTemporaries):
    """Perform type inference for finding the types of temporaries."""
    tmps = {tmp.id for tmp in node.tmps}
    domains: dict[str, ir.Expr] = {
        closure.output.id: closure.domain
        for closure in node.fencil.closures
        if closure.output.id in tmps
    }

    def convert_type(dtype):
        if isinstance(dtype, type_inference.Primitive):
            return dtype.name
        if isinstance(dtype, type_inference.TypeVar):
            return dtype.idx
        assert isinstance(dtype, type_inference.Tuple)
        dtypes = []
        while isinstance(dtype, type_inference.Tuple):
            dtypes.append(convert_type(dtype.front))
            dtype = dtype.others
        return tuple(dtypes)

    fencil_type = type_inference.infer(node.fencil)
    assert isinstance(fencil_type, type_inference.FencilDefinitionType)
    assert isinstance(fencil_type.params, type_inference.Tuple)
    all_types = []
    types = dict[str, ir.Expr]()
    for param, dtype in zip(node.fencil.params, fencil_type.params):
        assert isinstance(dtype, type_inference.Val)
        all_types.append(convert_type(dtype.dtype))
        if param.id in tmps:
            assert param.id not in types
            t = all_types[-1]
            types[param.id] = all_types.index(t) if isinstance(t, int) else t

    return FencilWithTemporaries(
        fencil=node.fencil,
        params=node.params,
        tmps=[
            Temporary(id=tmp.id, domain=domains[tmp.id], dtype=types[tmp.id]) for tmp in node.tmps
        ],
    )


class CreateGlobalTmps(NodeTranslator):
    """Main entry point for introducing global temporaries.

    Transforms an existing iterator IR fencil into a fencil with global temporaries.
    """

    def visit_FencilDefinition(
        self, node: ir.FencilDefinition, *, offset_provider
    ) -> FencilWithTemporaries:
        # Split closures on lifted function calls and introduce temporaries
        res = split_closures(node)
        # Prune unreferences closure inputs introduced in the previous step
        res = PruneClosureInputs().visit(res)
        # Perform an eta-reduction which should put all calls at the highest level of a closure
        res = EtaReduction().visit(res)
        # Perform a naive extent analysis to compute domain sizes of closures and temporaries
        if all(isinstance(o, CartesianAxis) for o in offset_provider.values()):
            res = update_cartesian_domains(res, offset_provider)
        else:
            res = update_unstructured_domains(res, offset_provider)
        # Use type inference to determine the data type of the temporaries
        return collect_tmps_info(res)
