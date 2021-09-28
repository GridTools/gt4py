# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Set, Union

from devtools import debug  # noqa: F401

import eve
from eve.iterators import TreeIterationItem
from gtc import common, oir
from gtc.common import CartesianOffset
from gtc.gtcpp import gtcpp


# - Each HorizontalExecution is a Functor (and a Stage)
# - Each VerticalLoop is MultiStage


def _extract_accessors(apply_method: gtcpp.GTApplyMethod) -> List[gtcpp.GTAccessor]:
    @eve.utils.as_xiter
    def _iter_valid_accessors(
        apply_method: gtcpp.GTApplyMethod,
    ) -> Generator[TreeIterationItem, None, None]:
        for stmt in apply_method.body:
            if not (
                isinstance(stmt, gtcpp.IfStmt)
                and stmt.cond.iter_tree()
                .if_isinstance(gtcpp.AccessorRef)
                .filter(lambda acc: acc.name[2:] == "pos")
                .to_list()
            ):
                yield from stmt.iter_tree().if_isinstance(gtcpp.AccessorRef)

    extents = _iter_valid_accessors(apply_method).reduceby(
        (lambda extent, accessor_ref: extent + accessor_ref.offset),
        "name",
        init=gtcpp.GTExtent.zero(),
        as_dict=True,
    )

    accessor_names = {
        name: gtcpp.GTExtent.zero()
        for name in apply_method.iter_tree()
        .if_isinstance(gtcpp.AccessorRef)
        .getattr("name")
        .to_set()
    }
    extents = {**{name: gtcpp.GTExtent.zero() for name in accessor_names}, **extents}

    inout_fields: Set[str] = (
        apply_method.iter_tree()
        .if_isinstance(gtcpp.AssignStmt)
        .getattr("left")
        .if_isinstance(gtcpp.AccessorRef)
        .getattr("name")
        .to_set()
    )
    ndims = dict(
        apply_method.iter_tree()
        .if_isinstance(gtcpp.AccessorRef)
        .map(lambda accessor: (accessor.name, 3 + len(accessor.data_index)))
    )

    return [
        gtcpp.GTAccessor(
            name=name,
            id=i,
            intent=gtcpp.Intent.INOUT if name in inout_fields else gtcpp.Intent.IN,
            extent=extent,
            ndim=ndims[name],
        )
        for i, (name, extent) in enumerate(extents.items())
    ]


class OIRToGTCpp(eve.NodeTranslator):
    @dataclass
    class ProgramContext:
        functors: List[gtcpp.GTFunctor] = field(default_factory=list)

        def add_functor(self, functor: gtcpp.GTFunctor) -> "OIRToGTCpp.ProgramContext":
            self.functors.append(functor)
            return self

    @dataclass
    class GTComputationContext:
        temporaries: List[gtcpp.Temporary] = field(default_factory=list)
        arguments: Set[gtcpp.Arg] = field(default_factory=set)
        axis_indices: Dict[str, str] = field(default_factory=dict)
        axis_endpoints: Dict[str, str] = field(default_factory=dict)

        def add_temporaries(
            self, temporaries: List[gtcpp.Temporary]
        ) -> "OIRToGTCpp.GTComputationContext":
            self.temporaries.extend(temporaries)
            return self

        def add_arguments(self, arguments: Set[gtcpp.Arg]) -> "OIRToGTCpp.GTComputationContext":
            self.arguments.update(arguments)
            return self

        def add_axis_index(self, axis: str) -> "OIRToGTCpp.GTComputationContext":
            self.axis_indices[axis] = f"{axis.lower()}_pos"
            return self

        def add_axis_endpoint(self, axis: str) -> "OIRToGTCpp.GTComputationContext":
            self.axis_endpoints[axis] = f"{axis.lower()}_length"
            return self

    contexts = (eve.SymbolTableTrait.symtable_merger,)

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> gtcpp.Literal:
        return gtcpp.Literal(value=node.value, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> gtcpp.UnaryOp:
        return gtcpp.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> gtcpp.BinaryOp:
        return gtcpp.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> gtcpp.TernaryOp:
        return gtcpp.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> gtcpp.NativeFuncCall:
        return gtcpp.NativeFuncCall(func=node.func, args=self.visit(node.args, **kwargs))

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> gtcpp.Cast:
        return gtcpp.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_Temporary(self, node: oir.Temporary, **kwargs: Any) -> gtcpp.Temporary:
        return gtcpp.Temporary(name=node.name, dtype=node.dtype)

    def visit_CartesianOffset(
        self, node: common.CartesianOffset, **kwargs: Any
    ) -> common.CartesianOffset:
        return node

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> gtcpp.AccessorRef:
        return gtcpp.AccessorRef(
            name=node.name,
            offset=self.visit(node.offset),
            data_index=node.data_index,
            dtype=node.dtype,
        )

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, **kwargs: Any
    ) -> Union[gtcpp.AccessorRef, gtcpp.LocalAccess]:
        assert "symtable" in kwargs
        if node.name in kwargs["symtable"]:
            symbol = kwargs["symtable"][node.name]
            if isinstance(symbol, oir.ScalarDecl):
                return gtcpp.AccessorRef(
                    name=symbol.name, offset=CartesianOffset.zero(), dtype=symbol.dtype
                )
            assert isinstance(symbol, oir.LocalScalar)
        return gtcpp.LocalAccess(name=node.name, dtype=node.dtype)

    def visit_AxisBound(
        self, node: oir.AxisBound, *, is_start: bool, **kwargs: Any
    ) -> gtcpp.GTLevel:
        if node.level == common.LevelMarker.START:
            splitter = 0
            offset = node.offset + 1 if (node.offset >= 0 and is_start) else node.offset
        elif node.level == common.LevelMarker.END:
            splitter = 1
            offset = node.offset - 1 if (node.offset <= 0 and not is_start) else node.offset
        else:
            raise ValueError("Cannot handle dynamic levels")
        return gtcpp.GTLevel(splitter=splitter, offset=offset)

    def visit_Interval(self, node: oir.Interval, **kwargs: Any) -> gtcpp.GTInterval:
        return gtcpp.GTInterval(
            from_level=self.visit(node.start, is_start=True),
            to_level=self.visit(node.end, is_start=False),
        )

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> gtcpp.AssignStmt:
        assert "symtable" in kwargs
        return gtcpp.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def _ref_from_axis_bound(
        self, axis_bound: common.AxisBound, *, axis: str, comp_ctx: GTComputationContext
    ) -> gtcpp.Expr:
        if axis_bound.level == common.LevelMarker.END:
            comp_ctx.add_axis_endpoint(axis)
            return gtcpp.BinaryOp(
                op=common.ArithmeticOperator.ADD,
                left=gtcpp.AccessorRef(
                    name=comp_ctx.axis_endpoints[axis],
                    offset=common.CartesianOffset.zero(),
                    dtype=common.DataType.INT32,
                ),
                right=gtcpp.Literal(value=str(axis_bound.offset), dtype=common.DataType.INT32),
            )
        else:
            return gtcpp.Literal(value=str(axis_bound.offset), dtype=common.DataType.INT32)

    def _expr_from_horizontal_interval(
        self,
        interval: common.HorizontalInterval,
        *,
        axis: str,
        comp_ctx: GTComputationContext,
        **kwargs: Any,
    ) -> gtcpp.Expr:
        comp_ctx.add_axis_index(axis)
        if interval.is_single_index:
            return gtcpp.BinaryOp(
                op=common.ComparisonOperator.EQ,
                left=gtcpp.AccessorRef(
                    name=comp_ctx.axis_indices[axis],
                    offset=common.CartesianOffset.zero(),
                    dtype=common.DataType.INT32,
                ),
                right=self._ref_from_axis_bound(interval.start, axis=axis, comp_ctx=comp_ctx),
            )
        else:
            if interval.start:
                start_expr = gtcpp.BinaryOp(
                    op=common.ComparisonOperator.GE,
                    left=gtcpp.AccessorRef(
                        name=comp_ctx.axis_indices[axis],
                        offset=common.CartesianOffset.zero(),
                        dtype=common.DataType.INT32,
                    ),
                    right=self._ref_from_axis_bound(interval.start, axis=axis, comp_ctx=comp_ctx),
                )
            else:
                start_expr = None

            if interval.end:
                end_expr = gtcpp.BinaryOp(
                    op=common.ComparisonOperator.LT,
                    left=gtcpp.AccessorRef(
                        name=comp_ctx.axis_indices[axis],
                        offset=common.CartesianOffset.zero(),
                        dtype=common.DataType.INT32,
                    ),
                    right=self._ref_from_axis_bound(interval.end, axis=axis, comp_ctx=comp_ctx),
                )
            else:
                end_expr = None

            if start_expr and end_expr:
                return gtcpp.BinaryOp(
                    op=common.LogicalOperator.AND, left=start_expr, right=end_expr
                )
            else:
                # Return the first non-None expr, or if all are None, then return None
                return next((expr for expr in (start_expr, end_expr) if expr is not None), None)

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs: Any) -> gtcpp.Expr:
        i_expr = self._expr_from_horizontal_interval(node.i, axis="I", **kwargs)
        j_expr = self._expr_from_horizontal_interval(node.j, axis="J", **kwargs)
        if i_expr and j_expr:
            return gtcpp.BinaryOp(op=common.LogicalOperator.AND, left=i_expr, right=j_expr)
        else:
            true_value = common.Literal(
                value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL
            )
            # Return the first non-None expr, or if all are None, then return True
            return next((expr for expr in (i_expr, j_expr) if expr is not None), true_value)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> gtcpp.IfStmt:
        return gtcpp.IfStmt(
            cond=self.visit(node.mask, **kwargs),
            true_branch=gtcpp.BlockStmt(body=self.visit(node.body, **kwargs)),
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        prog_ctx: ProgramContext,
        comp_ctx: GTComputationContext,
        interval: gtcpp.GTInterval,
        **kwargs: Any,
    ) -> gtcpp.GTStage:
        assert "symtable" in kwargs
        apply_method = gtcpp.GTApplyMethod(
            interval=self.visit(interval, **kwargs),
            body=self.visit(node.body, comp_ctx=comp_ctx, **kwargs),
            local_variables=self.visit(node.declarations, **kwargs),
        )
        accessors = _extract_accessors(apply_method)
        stage_args = [gtcpp.Arg(name=acc.name) for acc in accessors]

        comp_ctx.add_arguments(
            {
                param_arg
                for param_arg in stage_args
                if param_arg.name not in [tmp.name for tmp in comp_ctx.temporaries]
            }
        )

        functor_name = type(node).__name__ + str(id(node))
        prog_ctx.add_functor(
            gtcpp.GTFunctor(
                name=functor_name,
                applies=[apply_method],
                param_list=gtcpp.GTParamList(accessors=accessors),
            )
        ),

        return gtcpp.GTStage(functor=functor_name, args=stage_args)

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        comp_ctx: GTComputationContext,
        **kwargs: Any,
    ) -> gtcpp.GTMultiStage:
        # the following visit assumes that temporaries are already available in comp_ctx
        stages = list(
            itertools.chain(
                *(
                    self.visit(
                        section.horizontal_executions,
                        interval=section.interval,
                        default=([], []),
                        comp_ctx=comp_ctx,
                        **kwargs,
                    )
                    for section in node.sections
                )
            )
        )
        caches = self.visit(node.caches)
        return gtcpp.GTMultiStage(loop_order=node.loop_order, stages=stages, caches=caches)

    def visit_IJCache(self, node: oir.IJCache, **kwargs: Any) -> gtcpp.IJCache:
        return gtcpp.IJCache(name=node.name, loc=node.loc)

    def visit_KCache(self, node: oir.KCache, **kwargs: Any) -> gtcpp.KCache:
        return gtcpp.KCache(name=node.name, fill=node.fill, flush=node.flush, loc=node.loc)

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> gtcpp.FieldDecl:
        return gtcpp.FieldDecl(
            name=node.name, dtype=node.dtype, dimensions=node.dimensions, data_dims=node.data_dims
        )

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> gtcpp.GlobalParamDecl:
        return gtcpp.GlobalParamDecl(name=node.name, dtype=node.dtype)

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> gtcpp.LocalVarDecl:
        return gtcpp.LocalVarDecl(name=node.name, dtype=node.dtype, loc=node.loc)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> gtcpp.Program:
        prog_ctx = self.ProgramContext()
        comp_ctx = self.GTComputationContext()

        assert all([isinstance(decl, oir.Temporary) for decl in node.declarations])
        comp_ctx.add_temporaries(self.visit(node.declarations))

        multi_stages = self.visit(
            node.vertical_loops,
            prog_ctx=prog_ctx,
            comp_ctx=comp_ctx,
            **kwargs,
        )

        bindings = [
            gtcpp.Binding(name=name, expr=gtcpp.Positional(dim=axis.lower()))
            for axis, name in comp_ctx.axis_indices.items()
        ] + [
            gtcpp.Binding(name=name, expr=gtcpp.AxisEndpoint(axis={"I": 0, "J": 1, "K": 2}[axis]))
            for axis, name in comp_ctx.axis_endpoints.items()
        ]

        gt_computation = gtcpp.GTComputationCall(
            arguments=comp_ctx.arguments,
            extra_decls=bindings,
            temporaries=comp_ctx.temporaries,
            multi_stages=multi_stages,
        )
        parameters = self.visit(node.params)
        return gtcpp.Program(
            name=node.name,
            parameters=parameters,
            functors=prog_ctx.functors,
            gt_computation=gt_computation,
        )
