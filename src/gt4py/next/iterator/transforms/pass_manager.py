# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from typing import Optional, Protocol

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator import pretty_printer
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import (
    concat_where,
    dead_code_elimination,
    fuse_as_fieldop,
    global_tmps,
    infer_domain,
    infer_domain_ops,
    inline_dynamic_shifts,
    inline_fundefs,
    inline_lifts,
    prune_empty_concat_where,
    remove_broadcast,
)
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_scalar import InlineScalar
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.transforms.unroll_cartesian_reduce import UnrollCartesianReduce
from gt4py.next.iterator.type_system.inference import infer


class GTIRTransform(Protocol):
    def __call__(
        self, _: itir.Program, *, offset_provider: common.OffsetProvider
    ) -> itir.Program: ...


class _FieldviewDebugStats(eve.NodeVisitor):
    def __init__(self) -> None:
        self.cartesian_reduce_nodes: list[itir.FunCall] = []
        self.nested_as_fieldop_nodes: list[itir.FunCall] = []
        self.named_range_nodes: list[itir.FunCall] = []
        self.named_range_arg_debug: list[tuple[str, str, str]] = []
        self.named_range_count = 0

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if cpm.is_call_to(node, "cartesian_reduce"):
            self.cartesian_reduce_nodes.append(node)
        if cpm.is_call_to(node, "named_range"):
            self.named_range_count += 1
            self.named_range_nodes.append(node)
            axis, start, stop = node.args
            self.named_range_arg_debug.append(
                (
                    f"{type(axis).__name__}:{getattr(axis, 'value', None)}",
                    f"{type(start).__name__}:{getattr(start, 'value', getattr(start, 'name', None))}",
                    f"{type(stop).__name__}:{getattr(stop, 'value', getattr(stop, 'name', None))}",
                )
            )

        if cpm.is_applied_as_fieldop(node):
            if any(cpm.is_applied_as_fieldop(arg) for arg in node.args if isinstance(arg, itir.FunCall)):
                self.nested_as_fieldop_nodes.append(node)

        self.generic_visit(node, **kwargs)


def _debug_dump_fieldview_ir(stage: str, ir: itir.Program) -> None:
    if not os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR"):
        return

    stats = _FieldviewDebugStats()
    stats.visit(ir)

    print(
        f"[GT4PY_DEBUG_FIELDVIEW_IR] stage={stage} "
        f"cartesian_reduce_calls={len(stats.cartesian_reduce_nodes)} "
        f"nested_as_fieldop_calls={len(stats.nested_as_fieldop_nodes)} "
        f"named_range_calls={stats.named_range_count}",
        file=sys.stderr,
    )

    if stats.cartesian_reduce_nodes:
        print("[GT4PY_DEBUG_FIELDVIEW_IR] unresolved cartesian_reduce snippets:", file=sys.stderr)
        for node in stats.cartesian_reduce_nodes[:5]:
            print(pretty_printer.pformat(node), file=sys.stderr)

    if stats.nested_as_fieldop_nodes:
        print("[GT4PY_DEBUG_FIELDVIEW_IR] nested as_fieldop snippets:", file=sys.stderr)
        for node in stats.nested_as_fieldop_nodes[:5]:
            print(pretty_printer.pformat(node), file=sys.stderr)

    if stats.named_range_nodes:
        print("[GT4PY_DEBUG_FIELDVIEW_IR] named_range snippets:", file=sys.stderr)
        for node in stats.named_range_nodes[:5]:
            print(pretty_printer.pformat(node), file=sys.stderr)
        print("[GT4PY_DEBUG_FIELDVIEW_IR] named_range arg types:", file=sys.stderr)
        for axis_dbg, start_dbg, stop_dbg in stats.named_range_arg_debug[:5]:
            print(f"axis={axis_dbg} start={start_dbg} stop={stop_dbg}", file=sys.stderr)

    if os.environ.get("GT4PY_DEBUG_FIELDVIEW_IR_FULL"):
        print("[GT4PY_DEBUG_FIELDVIEW_IR] full pre-infer-domain IR:", file=sys.stderr)
        print(pretty_printer.pformat(ir), file=sys.stderr)


def _apply_unroll_reduce_pipeline(
    ir: itir.Program,
    *,
    offset_provider_type: common.OffsetProviderType,
    uids: utils.IDGeneratorPool,
    use_offset_literal_index: bool = True,
) -> itir.Program:
    for _ in range(10):
        try:
            unrolled = UnrollReduce.apply(
            ir,
            offset_provider_type=offset_provider_type,
            uids=uids,
            use_offset_literal_index=use_offset_literal_index,
            )
            unrolled = CollapseListGet().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
            # this is required as nested neighbor reductions can contain lifts, e.g.,
            # `neighbors(V2Eₒ, ↑f(...))`
            unrolled = inline_lifts.InlineLifts().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
        except Exception as e:
            raise RuntimeError("Failed inside _apply_unroll_reduce_pipeline") from e
        if unrolled == ir:
            break
        ir = unrolled
    else:
        raise RuntimeError("Reduction unrolling failed.")
    return ir


# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `extract_temporaries` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Program,
    *,
    # TODO(havogt): should be replaced by `common.OffsetProviderType`, but global_tmps currently
    #  relies on runtime info or `symbolic_domain_sizes`.
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    extract_temporaries=False,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    #: A dictionary mapping axes names to their length. See :func:`infer_domain.infer_expr` for
    #: more details.
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
) -> itir.Program:
    assert isinstance(ir, itir.Program)

    offset_provider_type = common.offset_provider_to_type(offset_provider)

    uids = utils.IDGeneratorPool()

    # print("\n" + "="*60)
    # print("=== FINAL GTIR HANDED TO DACE BACKEND ===")
    # print("="*60)
    # print(ir)
    # print("="*60 + "\n")
    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = NormalizeShifts().visit(ir)

    # TODO(tehrengruber): Many iterator test contain lifts that need to be inlined, e.g.
    #  test_can_deref. We didn't notice previously as FieldOpFusion did this implicitly everywhere.
    ir = inline_lifts.InlineLifts().visit(ir)

    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = dead_code_elimination.dead_code_elimination(
        ir, uids=uids, offset_provider_type=offset_provider_type
    )  # domain inference does not support dead-code
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet
    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)

    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    ir = remove_broadcast.RemoveBroadcast.apply(ir)

    ir = concat_where.transform_to_as_fieldop(ir)

    for _ in range(10):
        inlined = ir

        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]  # always an itir.Program
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(
            inlined,
            enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
            uids=uids,
            offset_provider_type=offset_provider_type,
        )  # type: ignore[assignment]  # always an itir.Program
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
            inlined, uids=uids, offset_provider_type=offset_provider_type
        )

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider_type=offset_provider_type)
        ir = global_tmps.create_global_tmps(
            ir,
            offset_provider=offset_provider,
            symbolic_domain_sizes=symbolic_domain_sizes,
            uids=uids,
        )

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps(uids=uids).visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        ir = _apply_unroll_reduce_pipeline(
            ir,
            offset_provider_type=offset_provider_type,
            uids=uids,
        )

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir


def apply_fieldview_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    unroll_reduce: bool = False,
) -> itir.Program:
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    uids = utils.IDGeneratorPool()

    print_ir = False
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR BEFORE FIELDVIEW TRANSFORMS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")

    ir = inline_fundefs.InlineFundefs().visit(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER INLINING FUNDEFS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER PRUNING UNREFERENCED FUNDEFS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    # required for dead-code-elimination and `prune_empty_concat_where` pass
    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER EXPANDING CONCAT_WHERE TUPLE ARGS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = dead_code_elimination.dead_code_elimination(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER DEAD CODE ELIMINATION ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER INLINING DYNAMIC SHIFTS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = infer_domain_ops.InferDomainOps.apply(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER INFERRING DOMAIN OPS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = concat_where.canonicalize_domain_argument(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER CANONICALIZING CONCAT_WHERE DOMAIN ARGUMENTS ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER CONSTANT FOLDING ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = UnrollCartesianReduce.apply(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER UNROLLING CARTESIAN REDUCE ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = infer_domain.infer_program(ir, offset_provider=offset_provider)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER INFERRING DOMAIN ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER PRUNING EMPTY CONCAT_WHERE ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER REMOVING BROADCAST ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")
    print(f"Unrolling reduce: {unroll_reduce}")
    if unroll_reduce:
        ir = _apply_unroll_reduce_pipeline(
            ir,
            offset_provider_type=offset_provider_type,
            uids=uids,
            use_offset_literal_index=False,  # Fieldview does not support non-literal offsets
        )
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(
            ir,
            opcount_preserving=True,
            force_inline_lambda_args=True,
        )
        ir = NormalizeShifts().visit(ir)

    
    if print_ir:
        print("\n" + "="*60)
        print("=== GTIR AFTER UNROLLING REDUCE (IF ENABLED) ===")
        print("="*60)
        print(ir)
        print("="*60 + "\n")

    return ir
