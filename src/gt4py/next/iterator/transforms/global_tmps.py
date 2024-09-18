# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Mapping
from typing import Any, Callable, Final, Iterable, Literal, Optional, Sequence

import gt4py.next as gtx
from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Dict, Tuple
from gt4py.eve.traits import SymbolTableTrait
from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.pretty_printer import PrettyPrinter
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.cse import extract_subexpression
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.prune_closure_inputs import PruneClosureInputs
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py.next.iterator.type_system import (
    inference as itir_type_inference,
    type_specifications as it_ts,
)
from gt4py.next.type_system import type_specifications as ts


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


AUTO_DOMAIN: Final = ir.FunCall(fun=ir.SymRef(id="_gtmp_auto_domain"), args=[])


# Iterator IR extension nodes


class FencilWithTemporaries(
    ir.Node, SymbolTableTrait
):  # TODO(havogt): remove and use new `itir.Program` instead.
    """Iterator IR extension: declaration of a fencil with temporary buffers."""

    fencil: ir.FencilDefinition
    params: list[ir.Sym]
    tmps: list[ir.Temporary]


# Extensions for `PrettyPrinter` for easier debugging


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


PrettyPrinter.visit_FencilWithTemporaries = pformat_FencilWithTemporaries  # type: ignore


# Main implementation
def canonicalize_applied_lift(closure_params: list[str], node: ir.FunCall) -> ir.FunCall:
    """
    Canonicalize applied lift expressions.

    Transform lift such that the arguments to the applied lift are only symbols.

    >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    >>> it_type = it_ts.IteratorType(position_dims=[], defined_dims=[], element_type=bool_type)
    >>> expr = im.lift(im.lambda_("a")(im.deref("a")))(im.lift("deref")(im.ref("inp", it_type)))
    >>> print(expr)
    (↑(λ(a) → ·a))((↑deref)(inp))
    >>> print(canonicalize_applied_lift(["inp"], expr))
    (↑(λ(inp) → (λ(a) → ·a)((↑deref)(inp))))(inp)
    """
    assert cpm.is_applied_lift(node)
    stencil = node.fun.args[0]  # type: ignore[attr-defined]  # ensured by is_applied lift
    it_args = node.args
    if any(not isinstance(it_arg, ir.SymRef) for it_arg in it_args):
        closure_param_refs = collect_symbol_refs(node, as_ref=True)
        assert not ({str(ref.id) for ref in closure_param_refs} - set(closure_params))
        new_node = im.lift(
            im.lambda_(*[im.sym(param.id) for param in closure_param_refs])(
                im.call(stencil)(*it_args)
            )
        )(*closure_param_refs)
        # ensure all types are inferred
        return itir_type_inference.infer(
            new_node, inplace=True, allow_undeclared_symbols=True, offset_provider={}
        )
    return node


@dataclasses.dataclass(frozen=True)
class TemporaryExtractionPredicate:
    """
    Construct a callable that determines if a lift expr can and should be extracted to a temporary.

    The class optionally takes a heuristic that can restrict the extraction.
    """

    heuristics: Optional[Callable[[ir.Expr], bool]] = None

    def __call__(self, expr: ir.Expr, num_occurences: int) -> bool:
        """Determine if `expr` is an applied lift that should be extracted as a temporary."""
        if not cpm.is_applied_lift(expr):
            return False
        # do not extract when the result is a list (i.e. a lift expression used in a `reduce` call)
        # as we can not create temporaries for these stencils
        assert isinstance(expr.type, it_ts.IteratorType)
        if isinstance(expr.type.element_type, it_ts.ListType):
            return False
        if self.heuristics and not self.heuristics(expr):
            return False
        stencil = expr.fun.args[0]  # type: ignore[attr-defined] # ensured by `is_applied_lift`
        # do not extract when the stencil is capturing
        used_symbols = collect_symbol_refs(stencil)
        if used_symbols:
            return False
        return True


@dataclasses.dataclass(frozen=True)
class SimpleTemporaryExtractionHeuristics:
    """
    Heuristic that extracts only if a lift expr is derefed in more than one position.

    Note that such expression result in redundant computations if inlined instead of being
    placed into a temporary.
    """

    closure: ir.StencilClosure

    def __post_init__(self) -> None:
        trace_shifts.trace_stencil(
            self.closure.stencil, num_args=len(self.closure.inputs), save_to_annex=True
        )

    def __call__(self, expr: ir.Expr) -> bool:
        shifts = expr.annex.recorded_shifts
        if len(shifts) > 1:
            return True
        return False


def _closure_parameter_argument_mapping(closure: ir.StencilClosure) -> dict[str, ir.Expr]:
    """
    Create a mapping from the closures parameters to the closure arguments.

    E.g. for the closure `out ← (λ(param) → ...)(arg) @ u⟨ ... ⟩;` we get a mapping from `param`
    to `arg`. In case the stencil is a scan, a mapping from closure inputs to scan pass (i.e. first
    arg is ignored) is returned.
    """
    is_scan = cpm.is_call_to(closure.stencil, "scan")

    if is_scan:
        stencil = closure.stencil.args[0]  # type: ignore[attr-defined]  # ensured by is_scan
        return {
            param.id: arg for param, arg in zip(stencil.params[1:], closure.inputs, strict=True)
        }
    else:
        assert isinstance(closure.stencil, ir.Lambda)
        return {
            param.id: arg for param, arg in zip(closure.stencil.params, closure.inputs, strict=True)
        }


def _ensure_expr_does_not_capture(expr: ir.Expr, whitelist: list[ir.Sym]) -> None:
    used_symbol_refs = collect_symbol_refs(expr)
    assert not (set(used_symbol_refs) - {param.id for param in whitelist})


def split_closures(
    node: ir.FencilDefinition,
    offset_provider: common.OffsetProvider,
    *,
    extraction_heuristics: Optional[
        Callable[[ir.StencilClosure], Callable[[ir.Expr], bool]]
    ] = None,
) -> FencilWithTemporaries:
    """Split closures on lifted function calls and introduce new temporary buffers for return values.

    Newly introduced temporaries will have the symbolic size of `AUTO_DOMAIN`. A symbol with the
    same name is also added as a fencil argument (to be replaced at a later stage).

    For each closure, follows these steps:
    1. Pops up lifted function calls to the top of the expression tree.
    2. Introduce new temporary for the output.
    3. Extract lifted function class as new closures with the previously created temporary as output.
    The closures are processed in reverse order to properly respect the dependencies.
    """
    if not extraction_heuristics:
        # extract all (eligible) lifts
        def always_extract_heuristics(_: ir.StencilClosure) -> Callable[[ir.Expr], bool]:
            return lambda _: True

        extraction_heuristics = always_extract_heuristics

    uid_gen_tmps = UIDGenerator(prefix="_tmp")

    node = itir_type_inference.infer(node, offset_provider=offset_provider)

    tmps: list[tuple[str, ts.DataType]] = []

    closures: list[ir.StencilClosure] = []
    for closure in reversed(node.closures):
        closure_stack: list[ir.StencilClosure] = [closure]
        while closure_stack:
            current_closure: ir.StencilClosure = closure_stack.pop()

            if (
                isinstance(current_closure.stencil, ir.SymRef)
                and current_closure.stencil.id == "deref"
            ):
                closures.append(current_closure)
                continue

            is_scan: bool = cpm.is_call_to(current_closure.stencil, "scan")
            current_closure_stencil = (
                current_closure.stencil if not is_scan else current_closure.stencil.args[0]  # type: ignore[attr-defined]  # ensured by is_scan
            )

            extraction_predicate = TemporaryExtractionPredicate(
                extraction_heuristics(current_closure)
            )

            stencil_body, extracted_lifts, _ = extract_subexpression(
                current_closure_stencil.expr,
                extraction_predicate,
                uid_gen_tmps,
                once_only=True,
                deepest_expr_first=True,
            )

            if extracted_lifts:
                for tmp_sym, lift_expr in extracted_lifts.items():
                    # make sure the applied lift is not capturing anything except of closure params
                    _ensure_expr_does_not_capture(lift_expr, current_closure_stencil.params)

                    assert isinstance(lift_expr, ir.FunCall) and isinstance(
                        lift_expr.fun, ir.FunCall
                    )

                    # make sure the arguments to the applied lift are only symbols
                    if not all(isinstance(arg, ir.SymRef) for arg in lift_expr.args):
                        lift_expr = canonicalize_applied_lift(
                            [str(param.id) for param in current_closure_stencil.params], lift_expr
                        )
                    assert all(isinstance(arg, ir.SymRef) for arg in lift_expr.args)

                    # create a mapping from the closures parameters to the closure arguments
                    closure_param_arg_mapping = _closure_parameter_argument_mapping(current_closure)

                    # usually an ir.Lambda or scan
                    stencil: ir.Node = lift_expr.fun.args[0]  # type: ignore[attr-defined] # ensured by canonicalize_applied_lift

                    # allocate a new temporary
                    assert isinstance(stencil.type, ts.FunctionType)
                    assert isinstance(stencil.type.returns, ts.DataType)
                    tmps.append((tmp_sym.id, stencil.type.returns))

                    # create a new closure that executes the stencil of the applied lift and
                    # writes the result to the newly created temporary
                    closure_stack.append(
                        ir.StencilClosure(
                            domain=AUTO_DOMAIN,
                            stencil=stencil,
                            output=im.ref(tmp_sym.id),
                            inputs=[
                                closure_param_arg_mapping[param.id]  # type: ignore[attr-defined]
                                for param in lift_expr.args
                            ],
                            location=current_closure.location,
                        )
                    )

                new_stencil: ir.Lambda | ir.FunCall
                # create a new stencil where all applied lifts that have been extracted are
                # replaced by references to the respective temporary
                new_stencil = ir.Lambda(
                    params=current_closure_stencil.params + list(extracted_lifts.keys()),
                    expr=stencil_body,
                )
                # if we are extracting from an applied scan we have to wrap the scan pass again,
                #  i.e. transform `λ(state, ...) → ...` into `scan(λ(state, ...) → ..., ...)`
                if is_scan:
                    new_stencil = im.call("scan")(new_stencil, current_closure.stencil.args[1:])  # type: ignore[attr-defined] # ensure by is_scan
                # inline such that let statements which are just rebinding temporaries disappear
                new_stencil = InlineLambdas.apply(
                    new_stencil, opcount_preserving=True, force_inline_lift_args=False
                )
                # we're done with the current closure, add it back to the stack for further
                # extraction.
                closure_stack.append(
                    ir.StencilClosure(
                        domain=current_closure.domain,
                        stencil=new_stencil,
                        output=current_closure.output,
                        inputs=current_closure.inputs
                        + [ir.SymRef(id=sym.id) for sym in extracted_lifts.keys()],
                        location=current_closure.location,
                    )
                )
            else:
                closures.append(current_closure)

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params + [im.sym(name) for name, _ in tmps] + [im.sym(AUTO_DOMAIN.fun.id)],  # type: ignore[attr-defined]  # value is a global constant
            closures=list(reversed(closures)),
            location=node.location,
            implicit_domain=node.implicit_domain,
        ),
        params=node.params,
        tmps=[ir.Temporary(id=name, dtype=type_) for name, type_ in tmps],
    )


def prune_unused_temporaries(node: FencilWithTemporaries) -> FencilWithTemporaries:
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
            location=node.fencil.location,
        ),
        params=node.params,
        tmps=[tmp for tmp in node.tmps if tmp.id not in unused_tmps],
    )


def _max_domain_sizes_by_location_type(offset_provider: Mapping[str, Any]) -> dict[str, int]:
    """Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `origin_axis` and the maximum
    value inside the neighbor table to get the size of each `neighbor_axis`.
    """
    sizes = dict[str, int]()
    for provider in offset_provider.values():
        if isinstance(provider, gtx.NeighborTableOffsetProvider):
            assert provider.origin_axis.kind == gtx.DimensionKind.HORIZONTAL
            assert provider.neighbor_axis.kind == gtx.DimensionKind.HORIZONTAL
            sizes[provider.origin_axis.value] = max(
                sizes.get(provider.origin_axis.value, 0), provider.table.shape[0]
            )
            sizes[provider.neighbor_axis.value] = max(
                sizes.get(provider.neighbor_axis.value, 0),
                provider.table.max() + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )
    return sizes


@dataclasses.dataclass
class SymbolicRange:
    start: ir.Expr
    stop: ir.Expr

    def translate(self, distance: int) -> "SymbolicRange":
        return SymbolicRange(im.plus(self.start, distance), im.plus(self.stop, distance))


@dataclasses.dataclass
class SymbolicDomain:
    grid_type: Literal["unstructured_domain", "cartesian_domain"]
    ranges: dict[
        common.Dimension, SymbolicRange
    ]  # TODO(havogt): remove `AxisLiteral` by `Dimension` everywhere

    @classmethod
    def from_expr(cls, node: ir.Node) -> SymbolicDomain:
        assert isinstance(node, ir.FunCall) and node.fun in [
            im.ref("unstructured_domain"),
            im.ref("cartesian_domain"),
        ]

        ranges: dict[common.Dimension, SymbolicRange] = {}
        for named_range in node.args:
            assert (
                isinstance(named_range, ir.FunCall)
                and isinstance(named_range.fun, ir.SymRef)
                and named_range.fun.id == "named_range"
            )
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, ir.AxisLiteral)

            ranges[common.Dimension(value=axis_literal.value, kind=axis_literal.kind)] = (
                SymbolicRange(lower_bound, upper_bound)
            )
        return cls(node.fun.id, ranges)  # type: ignore[attr-defined]  # ensure by assert above

    def as_expr(self) -> ir.FunCall:
        converted_ranges: dict[common.Dimension | str, tuple[ir.Expr, ir.Expr]] = {
            key: (value.start, value.stop) for key, value in self.ranges.items()
        }
        return im.domain(self.grid_type, converted_ranges)

    def translate(
        self: SymbolicDomain,
        shift: Tuple[ir.OffsetLiteral, ...],
        offset_provider: Dict[str, common.Dimension],
    ) -> SymbolicDomain:
        dims = list(self.ranges.keys())
        new_ranges = {dim: self.ranges[dim] for dim in dims}
        if len(shift) == 0:
            return self
        if len(shift) == 2:
            off, val = shift
            assert isinstance(off.value, str) and isinstance(val.value, int)
            nbt_provider = offset_provider[off.value]
            if isinstance(nbt_provider, common.Dimension):
                current_dim = nbt_provider
                # cartesian offset
                new_ranges[current_dim] = SymbolicRange.translate(
                    self.ranges[current_dim], val.value
                )
            elif isinstance(nbt_provider, common.Connectivity):
                # unstructured shift
                # note: ugly but cheap re-computation, but should disappear
                horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)

                old_dim = nbt_provider.origin_axis
                new_dim = nbt_provider.neighbor_axis

                assert new_dim not in new_ranges or old_dim == new_dim

                # TODO(tehrengruber): Do we need symbolic sizes, e.g., for ICON?
                new_range = SymbolicRange(
                    im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                    im.literal(str(horizontal_sizes[new_dim.value]), ir.INTEGER_INDEX_BUILTIN),
                )
                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )
            else:
                raise AssertionError()
            return SymbolicDomain(self.grid_type, new_ranges)
        elif len(shift) > 2:
            return self.translate(shift[0:2], offset_provider).translate(shift[2:], offset_provider)
        else:
            raise AssertionError("Number of shifts must be a multiple of 2.")


def domain_union(domains: list[SymbolicDomain]) -> SymbolicDomain:
    """Return the (set) union of a list of domains."""
    new_domain_ranges = {}
    assert all(domain.grid_type == domains[0].grid_type for domain in domains)
    assert all(domain.ranges.keys() == domains[0].ranges.keys() for domain in domains)
    for dim in domains[0].ranges.keys():
        start = functools.reduce(
            lambda current_expr, el_expr: im.call("minimum")(current_expr, el_expr),
            [domain.ranges[dim].start for domain in domains],
        )
        stop = functools.reduce(
            lambda current_expr, el_expr: im.call("maximum")(current_expr, el_expr),
            [domain.ranges[dim].stop for domain in domains],
        )
        new_domain_ranges[dim] = SymbolicRange(start, stop)

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)


def _group_offsets(
    offset_literals: Sequence[ir.OffsetLiteral],
) -> Sequence[tuple[str, int | Literal[trace_shifts.Sentinel.ALL_NEIGHBORS]]]:
    tags = [tag.value for tag in offset_literals[::2]]
    offsets = [
        offset.value if isinstance(offset, ir.OffsetLiteral) else offset
        for offset in offset_literals[1::2]
    ]
    assert all(isinstance(tag, str) for tag in tags)
    assert all(
        isinstance(offset, int) or offset == trace_shifts.Sentinel.ALL_NEIGHBORS
        for offset in offsets
    )
    return zip(tags, offsets, strict=True)  # type: ignore[return-value] # mypy doesn't infer literal correctly


def update_domains(
    node: FencilWithTemporaries,
    offset_provider: Mapping[str, Any],
    symbolic_sizes: Optional[dict[str, str]],
) -> FencilWithTemporaries:
    horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)
    closures: list[ir.StencilClosure] = []
    domains = dict[str, ir.FunCall]()
    for closure in reversed(node.fencil.closures):
        if closure.domain == AUTO_DOMAIN:
            # every closure with auto domain should have a single out field
            assert isinstance(closure.output, ir.SymRef)

            if closure.output.id not in domains:
                raise NotImplementedError(f"Closure output '{closure.output.id}' is never used.")

            domain = domains[closure.output.id]

            closure = ir.StencilClosure(
                domain=copy.deepcopy(domain),
                stencil=closure.stencil,
                output=closure.output,
                inputs=closure.inputs,
                location=closure.location,
            )
        else:
            domain = closure.domain

        closures.append(closure)

        local_shifts = trace_shifts.trace_stencil(closure.stencil, num_args=len(closure.inputs))
        for param_sym, shift_chains in zip(closure.inputs, local_shifts):
            param = param_sym.id
            assert isinstance(param, str)
            consumed_domains: list[SymbolicDomain] = (
                [SymbolicDomain.from_expr(domains[param])] if param in domains else []
            )
            for shift_chain in shift_chains:
                consumed_domain = SymbolicDomain.from_expr(domain)
                for offset_name, offset in _group_offsets(shift_chain):
                    if isinstance(offset_provider[offset_name], gtx.Dimension):
                        # cartesian shift
                        dim = offset_provider[offset_name]
                        assert offset is not trace_shifts.Sentinel.ALL_NEIGHBORS
                        consumed_domain.ranges[dim] = consumed_domain.ranges[dim].translate(offset)
                    elif isinstance(offset_provider[offset_name], common.Connectivity):
                        # unstructured shift
                        nbt_provider = offset_provider[offset_name]
                        old_axis = nbt_provider.origin_axis
                        new_axis = nbt_provider.neighbor_axis

                        assert new_axis not in consumed_domain.ranges or old_axis == new_axis

                        if symbolic_sizes is None:
                            new_range = SymbolicRange(
                                im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                                im.literal(
                                    str(horizontal_sizes[new_axis.value]), ir.INTEGER_INDEX_BUILTIN
                                ),
                            )
                        else:
                            new_range = SymbolicRange(
                                im.literal("0", ir.INTEGER_INDEX_BUILTIN),
                                im.ref(symbolic_sizes[new_axis.value]),
                            )
                        consumed_domain.ranges = dict(
                            (axis, range_) if axis != old_axis else (new_axis, new_range)
                            for axis, range_ in consumed_domain.ranges.items()
                        )
                        # TODO(tehrengruber): Revisit. Somehow the order matters so preserve it.
                        consumed_domain.ranges = dict(
                            (axis, range_) if axis != old_axis else (new_axis, new_range)
                            for axis, range_ in consumed_domain.ranges.items()
                        )
                    else:
                        raise NotImplementedError()
                consumed_domains.append(consumed_domain)

            # compute the bounds of all consumed domains
            if consumed_domains:
                if all(
                    consumed_domain.ranges.keys() == consumed_domains[0].ranges.keys()
                    for consumed_domain in consumed_domains
                ):  # scalar otherwise
                    domains[param] = domain_union(consumed_domains).as_expr()

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.fencil.params[:-1],  # remove `_gtmp_auto_domain` param again
            closures=list(reversed(closures)),
            location=node.fencil.location,
            implicit_domain=node.fencil.implicit_domain,
        ),
        params=node.params,
        tmps=node.tmps,
    )


def _tuple_constituents(node: ir.Expr) -> Iterable[ir.Expr]:
    if cpm.is_call_to(node, "make_tuple"):
        for arg in node.args:
            yield from _tuple_constituents(arg)
    else:
        yield node


def collect_tmps_info(
    node: FencilWithTemporaries, *, offset_provider: common.OffsetProvider
) -> FencilWithTemporaries:
    """Perform type inference for finding the types of temporaries and sets the temporary size."""
    tmps = {tmp.id for tmp in node.tmps}
    domains: dict[str, ir.Expr] = {}
    for closure in node.fencil.closures:
        for output_field in _tuple_constituents(closure.output):
            assert isinstance(output_field, ir.SymRef)
            if output_field.id not in tmps:
                continue

            assert output_field.id not in domains or domains[output_field.id] == closure.domain
            domains[output_field.id] = closure.domain

    new_node = FencilWithTemporaries(
        fencil=node.fencil,
        params=node.params,
        tmps=[
            ir.Temporary(id=tmp.id, domain=domains[tmp.id], dtype=tmp.dtype) for tmp in node.tmps
        ],
    )
    # TODO(tehrengruber): type inference is only really needed to infer the types of the temporaries
    #  and write them to the params of the inner fencil. This should be cleaned up after we
    #  refactored the IR.
    return itir_type_inference.infer(new_node, offset_provider=offset_provider)


def validate_no_dynamic_offsets(node: ir.Node) -> None:
    """Vaidate we have no dynamic offsets, e.g. `shift(Ioff, deref(...))(...)`"""
    for call_node in node.walk_values().if_isinstance(ir.FunCall):
        assert isinstance(call_node, ir.FunCall)
        if cpm.is_call_to(call_node, "shift"):
            if any(not isinstance(arg, ir.OffsetLiteral) for arg in call_node.args):
                raise NotImplementedError("Dynamic offsets not supported in temporary pass.")


# TODO(tehrengruber): Add support for dynamic shifts (e.g. the distance is a symbol). This can be
#  tricky: For every lift statement that is dynamically shifted we can not compute bounds anymore
#  and hence also not extract as a temporary.
class CreateGlobalTmps(PreserveLocationVisitor, NodeTranslator):
    """Main entry point for introducing global temporaries.

    Transforms an existing iterator IR fencil into a fencil with global temporaries.
    """

    def visit_FencilDefinition(
        self,
        node: ir.FencilDefinition,
        *,
        offset_provider: Mapping[str, Any],
        extraction_heuristics: Optional[
            Callable[[ir.StencilClosure], Callable[[ir.Expr], bool]]
        ] = None,
        symbolic_sizes: Optional[dict[str, str]],
    ) -> FencilWithTemporaries:
        # Vaidate we have no dynamic offsets, e.g. `shift(Ioff, deref(...))(...)`
        validate_no_dynamic_offsets(node)
        # Split closures on lifted function calls and introduce temporaries
        res = split_closures(
            node, offset_provider=offset_provider, extraction_heuristics=extraction_heuristics
        )
        # Prune unreferences closure inputs introduced in the previous step
        res = PruneClosureInputs().visit(res)
        # Prune unused temporaries possibly introduced in the previous step
        res = prune_unused_temporaries(res)
        # Perform an eta-reduction which should put all calls at the highest level of a closure
        res = EtaReduction().visit(res)
        # Perform a naive extent analysis to compute domain sizes of closures and temporaries
        res = update_domains(res, offset_provider, symbolic_sizes)
        # Use type inference to determine the data type of the temporaries
        return collect_tmps_info(res, offset_provider=offset_provider)
