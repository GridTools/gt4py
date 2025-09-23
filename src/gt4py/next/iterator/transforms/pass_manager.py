# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Mapping, Optional, Protocol

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
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
    remove_broadcast,
    symbol_ref_utils,
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
from gt4py.next.iterator.type_system.inference import infer


class GTIRTransform(Protocol):
    def __call__(
        self, _: itir.Program, *, offset_provider: common.OffsetProvider
    ) -> itir.Program: ...


def _max_domain_range_sizes(offset_provider: Mapping[str, Any]) -> dict[str, itir.Literal]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `source_dim` and the maximum
    value inside the neighbor table to get the size of each `codomain`.
    """
    sizes: dict[str, int] = {}
    for provider in offset_provider.values():
        if common.is_neighbor_connectivity(provider):
            conn_type = provider.__gt_type__()
            sizes[conn_type.source_dim.value] = max(
                sizes.get(conn_type.source_dim.value, 0), provider.ndarray.shape[0]
            )
            sizes[conn_type.codomain.value] = max(
                sizes.get(conn_type.codomain.value, 0),
                provider.ndarray.max() + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )
    sizes_exprs = {k: im.literal_from_value(v) for k, v in sizes.items()}
    return sizes_exprs


def _has_dynamic_domains(ir: itir.Program) -> bool:
    # note: this function does not respect symbol collisions with builtins. As it is a temporary
    # workaround we don't care about this corner cases.
    domains = set()
    domains |= ir.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    for as_fop in (
        ir.walk_values()
        .if_isinstance(itir.FunCall)
        .filter(lambda node: cpm.is_call_to(node, "as_fieldop") and len(node.args) == 2)
    ):
        domains.add(as_fop.args[1])
    return len(symbol_ref_utils.collect_symbol_refs(domains)) > 0


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
    symbolic_domain_sizes: Optional[dict[str, str | itir.Expr]] = None,
) -> itir.Program:
    assert isinstance(ir, itir.Program)

    offset_provider_type = common.offset_provider_to_type(offset_provider)

    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    if _has_dynamic_domains(ir):
        assert not symbolic_domain_sizes, "Options are mutually exclusive."
        symbolic_domain_sizes = _max_domain_range_sizes(offset_provider)  # type: ignore[assignment]

    tmp_uids = eve_utils.UIDGenerator(prefix="__tmp")
    mergeasfop_uids = eve_utils.UIDGenerator()
    collapse_tuple_uids = eve_utils.UIDGenerator()

    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = NormalizeShifts().visit(ir)

    # TODO(tehrengruber): Many iterator test contain lifts that need to be inlined, e.g.
    #  test_can_deref. We didn't notice previously as FieldOpFusion did this implicitly everywhere.
    ir = inline_lifts.InlineLifts().visit(ir)

    ir = dead_code_elimination.dead_code_elimination(
        ir, collapse_tuple_uids=collapse_tuple_uids, offset_provider_type=offset_provider_type
    )  # domain inference does not support dead-code
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir
    )  # domain inference does not support dynamic offsets yet
    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)

    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
    )
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
            uids=collapse_tuple_uids,
            offset_provider_type=offset_provider_type,
        )  # type: ignore[assignment]  # always an itir.Program
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
            inlined, uids=mergeasfop_uids, offset_provider_type=offset_provider_type
        )

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(ir, offset_provider_type=offset_provider_type)
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider_type=offset_provider_type)
        ir = global_tmps.create_global_tmps(
            ir,
            offset_provider=offset_provider,
            symbolic_domain_sizes=symbolic_domain_sizes,
            uids=tmp_uids,
        )

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps().visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider_type=offset_provider_type)
            unrolled = CollapseListGet().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
            # this is required as nested neighbor reductions can contain lifts, e.g.,
            # `neighbors(V2Eₒ, ↑f(...))`
            unrolled = inline_lifts.InlineLifts().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
            if unrolled == ir:
                break
            ir = unrolled
        else:
            raise RuntimeError("Reduction unrolling failed.")

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir


def apply_fieldview_transforms(
    ir: itir.Program, *, offset_provider: common.OffsetProvider
) -> itir.Program:
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    if _has_dynamic_domains(ir):
        symbolic_domain_sizes = _max_domain_range_sizes(offset_provider)  # type: ignore[assignment]
    else:
        symbolic_domain_sizes = None

    ir = inline_fundefs.InlineFundefs().visit(ir)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = dead_code_elimination.dead_code_elimination(ir, offset_provider_type=offset_provider_type)
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir
    )  # domain inference does not support dynamic offsets yet

    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)
    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

    ir = infer_domain.infer_program(
        ir, symbolic_domain_sizes=symbolic_domain_sizes, offset_provider=offset_provider
    )
    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    return ir
