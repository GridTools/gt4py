# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import copy
import enum
from typing import Callable, Optional

import dataclasses

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import fencil_to_program, infer_domain
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.global_tmps import CreateGlobalTmps, FencilWithTemporaries
from gt4py.next.iterator.transforms.inline_center_deref_lift_vars import InlineCenterDerefLiftVars
from gt4py.next.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from gt4py.next.iterator.transforms.inline_into_scan import InlineIntoScan
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas, inline_lambda
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref
from gt4py.next.iterator.transforms.scan_eta_reduction import ScanEtaReduction
from gt4py.next.iterator.transforms.trace_shifts import trace_stencil
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator.type_system.inference import infer
from gt4py.next.type_system import type_info
from gt4py.next.type_system.type_info import apply_to_primitive_constituents
from gt4py.next.iterator.type_system import type_specifications as ts


@enum.unique
class LiftMode(enum.Enum):
    FORCE_INLINE = enum.auto()
    USE_TEMPORARIES = enum.auto()


def _inline_lifts(ir, lift_mode):
    if lift_mode == LiftMode.FORCE_INLINE:
        return InlineLifts().visit(ir)
    elif lift_mode == LiftMode.USE_TEMPORARIES:
        return InlineLifts(
            flags=InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT
            | InlineLifts.Flag.INLINE_DEREF_LIFT  # some tuple exprs found in FVM don't work yet.
        ).visit(ir)
    else:
        raise ValueError()

    return ir


def _inline_into_scan(ir, *, max_iter=10):
    for _ in range(10):
        # in case there are multiple levels of lambdas around the scan we have to do multiple iterations
        inlined = InlineIntoScan().visit(ir)
        inlined = InlineLambdas.apply(inlined, opcount_preserving=True, force_inline_lift_args=True)
        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError(f"Inlining into 'scan' did not converge within {max_iter} iterations.")
    return ir

from gt4py.next.iterator.ir_utils import ir_makers as im

@dataclasses.dataclass
class MergeAsFieldOp(eve.NodeTranslator):
    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(cls, node: itir.Program, *, offset_provider, uids: Optional[eve_utils.UIDGenerator] = None):
        node = infer(node, offset_provider=offset_provider)
        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)

        if cpm.is_call_to(node.fun, "as_fieldop"):
            stencil = node.fun.args[0]
            domain = node.fun.args[1] if len(node.fun.args) > 1 else None
            if isinstance(stencil, itir.SymRef) and stencil.id == "deref":
                stencil = im.lambda_("arg")(im.deref("arg"))
            type_ = node.type
            node = im.as_fieldop(stencil, domain)(*node.args)
            node.type = type_

        if cpm.is_call_to(node.fun, "as_fieldop") and isinstance(node.fun.args[0], itir.Lambda):
            stencil = node.fun.args[0]
            domain = node.fun.args[1] if len(node.fun.args) > 1 else None

            shifts = trace_stencil(stencil)

            args = node.args

            new_stencil_body = stencil.expr
            new_stencil_params = []
            new_args = []
            duplicate_args_tracker = {}
            for stencil_param, arg, arg_shifts in zip(stencil.params, args, shifts, strict=True):
                dtype = type_info.extract_dtype(arg.type)
                should_inline = (
                    isinstance(arg, itir.FunCall) and
                    (cpm.is_call_to(arg.fun, "as_fieldop") or cpm.is_call_to(arg, "cond")) and
                    (isinstance(dtype, ts.ListType) or len(arg_shifts) <= 1)
                )
                if should_inline:
                    if cpm.is_call_to(arg.fun, "as_fieldop"):
                        pass
                    elif cpm.is_call_to(arg, "cond"):
                        type_ = arg.type
                        arg = im.op_as_fieldop("if_")(*arg.args)
                        arg.type = type_
                    else:
                        raise NotImplementedError()

                    inner_args = arg.args

                    new_stencil_params_for_arg = []
                    for inner_arg in inner_args:
                        if isinstance(inner_arg, itir.SymRef):
                            if inner_arg.id not in duplicate_args_tracker:
                                new_stencil_param = self.uids.sequential_id(prefix="__iasfop")
                                duplicate_args_tracker[inner_arg.id] = new_stencil_param
                                new_args.append(inner_arg)
                                new_stencil_params.append(new_stencil_param)
                            new_stencil_params_for_arg.append(duplicate_args_tracker[inner_arg.id])
                        else:
                            new_stencil_param = self.uids.sequential_id(prefix="__iasfop")
                            new_args.append(inner_arg)
                            new_stencil_params.append(new_stencil_param)
                            new_stencil_params_for_arg.append(new_stencil_param)

                    arg_stencil = arg.fun.args[0]
                    new_stencil_body = im.let(
                        stencil_param, im.lift(arg_stencil)(*new_stencil_params_for_arg)
                    )(
                        new_stencil_body
                    )
                else:
                    # see test_tuple_with_local_field_in_reduction_shifted for ex where assert fails
                    #assert not isinstance(dtype, ts.ListType)
                    new_stencil_params.append(stencil_param.id)
                    new_args.append(arg)

            new_stencil_body = InlineLambdas.apply(
                new_stencil_body,
                opcount_preserving=True,
                force_inline_lift_args=False,
                # If trivial lifts are not inlined we might create temporaries for constants. In all
                #  other cases we want it anyway.
                force_inline_trivial_lift_args=True,
            )
            new_stencil_body = InlineLifts().visit(new_stencil_body)

            new_node = im.as_fieldop(im.lambda_(*new_stencil_params)(new_stencil_body), domain)(*new_args)
            new_node.type = node.type
            return new_node
        return node

@dataclasses.dataclass(frozen=True)
class CreateGlobalTmps2(eve.NodeTranslator):
    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(cls, node: itir.Program, uids: Optional[eve_utils.UIDGenerator] = None):
        if not uids:
            uids = eve_utils.UIDGenerator()

        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        declarations = kwargs["declarations"]
        new_body = kwargs["new_body"]

        # this has to be run before the generic visit, otherwise we create undefined symbols
        if cpm.is_let(node):
            new_var_values = [
                self.visit(arg, declarations=declarations, new_body=new_body) for arg in node.args
            ]

            # TODO: double check expr duplication
            node = inline_lambda(im.let(*zip(node.fun.params, new_var_values))(node.fun.expr))

        node = self.generic_visit(node, **kwargs)

        if cpm.is_call_to(node, "as_fieldop"):  # don't look at local-view
            return node

        if cpm.is_call_to(node.fun, "as_fieldop"):
            tmp_name = self.uids.sequential_id(prefix="__tmp")
            stencil, domain = node.fun.args

            scalar_type = apply_to_primitive_constituents(
                type_info.extract_dtype, node.type
            )

            # TODO
            node = copy.deepcopy(node)
            node.args = self.visit(node.args, declarations=declarations, new_body=new_body)

            declarations.append(
                itir.Temporary(id=tmp_name, domain=domain, dtype=scalar_type)
            )
            new_body.append(
                itir.SetAt(
                    expr=node,
                    domain=domain,
                    target=im.ref(tmp_name)
                )
            )

            return im.ref(tmp_name)

        return node

    def visit_Program(self, node: itir.Program):
        declarations = node.declarations
        new_body = []

        for stmt in node.body:
            if isinstance(stmt, itir.SetAt):
                new_body.append(
                    itir.SetAt(
                        expr=self.visit(stmt.expr, declarations=declarations, new_body=new_body),
                        domain=stmt.domain,
                        target=stmt.target
                    )
                )
            else:
                raise NotImplementedError()

        return itir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            declarations=declarations,
            body=new_body
        )





# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `lift_mode` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    unconditionally_collapse_tuples=False,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
) -> itir.Program:
    if isinstance(ir, (itir.FencilDefinition, FencilWithTemporaries)):
        ir = fencil_to_program.FencilToProgram().apply(
            ir
        )  # FIXME[#1582](havogt): should be removed after refactoring to combined IR
    else:
        assert isinstance(ir, itir.Program)
        # FIXME[#1582](havogt): note: currently the case when using the roundtrip backend
        pass

    icdlv_uids = eve_utils.UIDGenerator()
    tmp_uids = eve_utils.UIDGenerator()
    mergeasfop_uids = eve_utils.UIDGenerator()

    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)

    # needs to run after infer_domain as req by type_inference
    #ir = MergeAsFieldOp.apply(ir, offset_provider=offset_provider)

    ir = MergeLet().visit(ir)
    ir = InlineFundefs().visit(ir)

    ir = PruneUnreferencedFundefs().visit(ir)
    ir = PropagateDeref.apply(ir)
    ir = NormalizeShifts().visit(ir)

    ir = infer_domain.infer_program(ir, offset_provider=offset_provider)

    for _ in range(10):
        inlined = ir

        inlined = MergeAsFieldOp.apply(inlined, uids=mergeasfop_uids, offset_provider=offset_provider)

        #inlined = InlineCenterDerefLiftVars.apply(inlined, uids=icdlv_uids)  # type: ignore[arg-type]  # always a fencil
        inlined = _inline_lifts(inlined, lift_mode)

        inlined = InlineLambdas.apply(
            inlined,
            opcount_preserving=True,
            force_inline_lift_args=(lift_mode == LiftMode.FORCE_INLINE),
            # If trivial lifts are not inlined we might create temporaries for constants. In all
            #  other cases we want it anyway.
            force_inline_trivial_lift_args=True,
        )
        inlined = ConstantFolding.apply(inlined)
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(
            inlined,
            offset_provider=offset_provider,
            # TODO(tehrengruber): disabled since it increases compile-time too much right now
            flags=~CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES,
        )
        # This pass is required such that a deref outside of a
        # `tuple_get(make_tuple(let(...), ...))` call is propagated into the let after the
        # `tuple_get` is removed by the `CollapseTuple` pass.
        inlined = PropagateDeref.apply(inlined)

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    if lift_mode != LiftMode.FORCE_INLINE:
        ir = infer(ir, inplace=True, offset_provider=offset_provider)
        ir = CreateGlobalTmps2.apply(ir, uids=tmp_uids)

        # FIXME[#1582](tehrengruber): implement new temporary pass here
        # raise NotImplementedError()
        # assert offset_provider is not None
        # ir = CreateGlobalTmps().visit(
        #     ir,
        #     offset_provider=offset_provider,
        #     extraction_heuristics=temporary_extraction_heuristics,
        #     symbolic_sizes=symbolic_domain_sizes,
        # )
        #
        # for _ in range(10):
        #     inlined = InlineLifts().visit(ir)
        #     inlined = InlineLambdas.apply(
        #         inlined, opcount_preserving=True, force_inline_lift_args=True
        #     )
        #     if inlined == ir:
        #         break
        #     ir = inlined
        # else:
        #     raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

        # If after creating temporaries, the scan is not at the top, we inline.
        # The following example doesn't have a lift around the shift, i.e. temporary pass will not extract it.
        # λ(inp) → scan(λ(state, k, kp) → state + ·k + ·kp, True, 0.0)(inp, ⟪Koffₒ, 1ₒ⟫(inp))`
        ir = _inline_into_scan(ir)

    # Since `CollapseTuple` relies on the type inference which does not support returning tuples
    # larger than the number of closure outputs as given by the unconditional collapse, we can
    # only run the unconditional version here instead of in the loop above.
    if unconditionally_collapse_tuples:
        ir = CollapseTuple.apply(
            ir,
            ignore_tuple_size=True,
            offset_provider=offset_provider,
            # TODO(tehrengruber): disabled since it increases compile-time too much right now
            flags=~CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES,
        )

    if lift_mode == LiftMode.FORCE_INLINE:
        ir = _inline_into_scan(ir)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps().visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider=offset_provider)
            if unrolled == ir:
                break
            ir = unrolled
            ir = CollapseListGet().visit(ir)
            ir = NormalizeShifts().visit(ir)
            ir = _inline_lifts(ir, LiftMode.FORCE_INLINE)
            ir = NormalizeShifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed.")

    ir = EtaReduction().visit(ir)
    ir = ScanEtaReduction().visit(ir)

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(ir, offset_provider=offset_provider)  # type: ignore[type-var]  # always an itir.Program
        ir = MergeLet().visit(ir)

    if lift_mode != LiftMode.FORCE_INLINE:
        ir = infer(ir, inplace=True, offset_provider=offset_provider)
        ir = CreateGlobalTmps2.apply(ir, uids=tmp_uids)

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir
