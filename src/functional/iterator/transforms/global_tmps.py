# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Optional

from eve import Coerced, NodeTranslator
from eve.traits import SymbolTableTrait
from functional.common import DimensionKind
from functional.iterator import ir, type_inference
from functional.iterator.embedded import NeighborTableOffsetProvider
from functional.iterator.pretty_printer import PrettyPrinter
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.popup_tmps import PopupTmps
from functional.iterator.transforms.prune_closure_inputs import PruneClosureInputs
from functional.iterator.transforms.trace_shifts import TraceShifts


AUTO_DOMAIN = ir.SymRef(id="_gtmp_auto_domain")


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


def prune_unused_temporaries(node: FencilWithTemporaries):
    """Remove temporaries that are never read."""
    unused_tmps = {tmp.id for tmp in node.tmps}
    for closure in node.fencil.closures:
        unused_tmps -= {inp.id for inp in closure.inputs}

    if not unused_tmps:
        return node

    closures = [
        closure
        for closure in node.fencil.closures
        if not (isinstance(closure.output, ir.SymRef) and closure.output.id in unused_tmps)
    ]
    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=[p for p in node.fencil.params if p.id not in unused_tmps],
            closures=closures,
        ),
        params=node.params,
        tmps=[tmp for tmp in node.tmps if tmp.id not in unused_tmps],
    )


def _offset_limits(offsets, offset_provider):
    offset_limits = {k: (0, 0) for k in offset_provider.keys()}
    for shift in offsets:
        offsets = {k: 0 for k in offset_provider.keys()}
        for k, v in zip(shift[0::2], shift[1::2]):
            assert isinstance(v, ir.OffsetLiteral) and isinstance(v.value, int)
            offsets[k.value] += v.value
        for k, v in offsets.items():
            old_min, old_max = offset_limits[k]
            offset_limits[k] = (min(old_min, v), max(old_max, v))

    return {v.value: offset_limits[k] for k, v in offset_provider.items()}


def _named_range_with_offsets(axis_literal, lower_bound, upper_bound, lower_offset, upper_offset):
    if lower_offset:
        lower_bound = ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[lower_bound, ir.Literal(value=str(lower_offset), type="int")],
        )
    if upper_offset:
        upper_bound = ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[upper_bound, ir.Literal(value=str(upper_offset), type="int")],
        )
    return ir.FunCall(
        fun=ir.SymRef(id="named_range"), args=[axis_literal, lower_bound, upper_bound]
    )


def _extend_cartesian_domain(domain, offsets, offset_provider):
    if not any(offsets):
        return domain
    assert isinstance(domain, ir.FunCall) and domain.fun == ir.SymRef(id="cartesian_domain")
    assert all(isinstance(axis, CartesianAxis) for axis in offset_provider.values())

    offset_limits = _offset_limits(offsets, offset_provider)

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
            _named_range_with_offsets(
                axis_literal, lower_bound, upper_bound, lower_offset, upper_offset
            )
        )

    return ir.FunCall(fun=domain.fun, args=named_ranges)


def update_cartesian_domains(node: FencilWithTemporaries, offset_provider) -> FencilWithTemporaries:
    """Replace appearances of `AUTO_DOMAIN` by concrete domain sizes.

    Naive extent analysis, does not handle boundary conditions etc. in a smart way.
    """
    closures = []
    domains = dict[str, ir.Expr]()
    for closure in reversed(node.fencil.closures):
        if closure.domain == AUTO_DOMAIN:
            domain = domains[closure.output.id]
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )
        else:
            domain = closure.domain

        closures.append(closure)

        if closure.stencil == ir.SymRef(id="deref"):
            domains[closure.inputs[0].id] = domain
            continue

        local_shifts = TraceShifts.apply(closure)
        for param, shifts in local_shifts.items():
            domains[param] = _extend_cartesian_domain(domain, shifts, offset_provider)

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


def _location_type_from_offsets(domain, offsets, offset_provider):
    location = domain.args[0].args[0].value
    for o in offsets:
        if isinstance(o, ir.OffsetLiteral) and isinstance(o.value, str):
            provider = offset_provider[o.value]
            if isinstance(provider, NeighborTableOffsetProvider):
                location = provider.neighbor_axis.value
    return location


def _unstructured_domain(axis, size, vertical_ranges):
    return ir.FunCall(
        fun=ir.SymRef(id="unstructured_domain"),
        args=[
            ir.FunCall(
                fun=ir.SymRef(id="named_range"),
                args=[
                    ir.AxisLiteral(value=axis),
                    ir.Literal(value="0", type="int"),
                    ir.Literal(value=str(size), type="int"),
                ],
            )
        ]
        + list(vertical_ranges),
    )


def _max_domain_sizes_by_location_type(offset_provider):
    sizes = dict[str, int]()
    for provider in offset_provider.values():
        if isinstance(provider, NeighborTableOffsetProvider):
            assert provider.origin_axis.kind == DimensionKind.HORIZONTAL
            assert provider.neighbor_axis.kind == DimensionKind.HORIZONTAL
            sizes[provider.origin_axis.value] = max(
                sizes.get(provider.origin_axis.value, 0), provider.tbl.shape[0]
            )
            sizes[provider.neighbor_axis.value] = max(
                sizes.get(provider.neighbor_axis.value, 0),
                provider.tbl.max(),
            )
    return sizes


def _domain_ranges(closures, offset_provider):
    ranges = dict[str, set[ir.Expr]]()
    for closure in closures:
        domain = closure.domain
        if isinstance(domain, ir.FunCall) and domain.fun == ir.SymRef(id="unstructured_domain"):
            for arg in domain.args:
                assert isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="named_range")
                axis = arg.args[0].value
                ranges.setdefault(axis, []).append(arg)
    return ranges


def update_unstructured_domains(node: FencilWithTemporaries, offset_provider):
    """Replace appearances of `AUTO_DOMAIN` by concrete domain sizes."""
    horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)
    vertical_ranges = _domain_ranges(node.fencil.closures, offset_provider)
    for k in horizontal_sizes:
        vertical_ranges.pop(k, None)

    closures = []
    domains = dict[str, ir.Expr]()
    for closure in reversed(node.fencil.closures):
        if closure.domain == AUTO_DOMAIN:
            domain = domains[closure.output.id]
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )
        else:
            domain = closure.domain

        closures.append(closure)

        if closure.stencil == ir.SymRef(id="deref"):
            domains[closure.inputs[0].id] = domain
            continue

        local_shifts = TraceShifts.apply(closure)
        for param, shifts in local_shifts.items():
            loctypes = {_location_type_from_offsets(domain, s, offset_provider) for s in shifts}
            assert len(loctypes) == 1
            loctype = loctypes.pop()
            horizontal_size = horizontal_sizes[loctype]
            domains[param] = _unstructured_domain(
                loctype, horizontal_size, vertical_ranges.values()
            )

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


def collect_tmps_info(node: FencilWithTemporaries):
    """Perform type inference for finding the types of temporaries and sets the temporary size."""
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
    def visit_FencilDefinition(
        self, node: ir.FencilDefinition, *, offset_provider
    ) -> FencilWithTemporaries:
        # Split closures on lifted function calls and introduce temporaries
        res = split_closures(node)
        # Prune unreferences closure inputs introduced in the previous step
        res = PruneClosureInputs().visit(res)
        # Prune unused temporaries possibly introduced in the previous step
        res = prune_unused_temporaries(res)
        # Perform an eta-reduction which should put all calls at the highest level of a closure
        res = EtaReduction().visit(res)
        # Perform a naive extent analysis to compute domain sizes of closures and temporaries
        if all(isinstance(o, CartesianAxis) for o in offset_provider.values()):
            res = update_cartesian_domains(res, offset_provider)
        else:
            res = update_unstructured_domains(res, offset_provider)
        # Use type inference to determine the data type of the temporaries
        return collect_tmps_info(res)
