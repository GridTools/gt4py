# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import functools
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Union, cast

from devtools import debug  # noqa: F401
from typing_extensions import Protocol

import eve
from gtc import common, oir
from gtc.common import CartesianOffset, ExprKind
from gtc.gtcpp import gtcpp
from gtc.passes.oir_optimizations.utils import collect_symbol_names, symbol_name_creator


# - Each HorizontalExecution is a Functor (and a Stage)
# - Each VerticalLoop is MultiStage


def _extract_accessors(node: eve.Node, temp_names: Set[str]) -> List[gtcpp.GTAccessor]:
    extents = (
        node.walk_values()
        .if_isinstance(gtcpp.AccessorRef)
        .reduceby(
            (lambda extent, accessor_ref: extent + accessor_ref.offset),
            "name",
            init=gtcpp.GTExtent.zero(),
            as_dict=True,
        )
    )

    inout_fields: Set[str] = (
        node.walk_values()
        .if_isinstance(gtcpp.AssignStmt)
        .getattr("left")
        .if_isinstance(gtcpp.AccessorRef)
        .getattr("name")
        .to_set()
    )
    ndims = dict(
        node.walk_values()
        .if_isinstance(gtcpp.AccessorRef)
        .map(
            lambda accessor: (
                accessor.name,
                3 + (len(accessor.data_index) if accessor.name not in temp_names else 0),
            )
        )
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


def _make_axis_offset_expr(
    bound: common.AxisBound,
    axis_index: int,
    axis_length_accessor: Callable[[int], gtcpp.AccessorRef],
) -> gtcpp.Expr:
    if bound.level == common.LevelMarker.END:
        base = axis_length_accessor(axis_index)
        return gtcpp.BinaryOp(
            op=common.ArithmeticOperator.ADD,
            left=base,
            right=gtcpp.Literal(value=str(bound.offset), dtype=common.DataType.INT32),
        )
    else:
        return gtcpp.Literal(value=str(bound.offset), dtype=common.DataType.INT32)


class SymbolNameCreator(Protocol):
    def __call__(self, name: str) -> str:
        ...


class OIRToGTCpp(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    @dataclass
    class ProgramContext:
        functors: List[gtcpp.GTFunctor] = field(default_factory=list)

        def add_functor(self, functor: gtcpp.GTFunctor) -> "OIRToGTCpp.ProgramContext":
            self.functors.append(functor)
            return self

    @dataclass
    class GTComputationContext:
        create_symbol_name: SymbolNameCreator
        temporaries: List[gtcpp.Temporary] = field(default_factory=list)
        positionals: Dict[int, gtcpp.Positional] = field(default_factory=dict)
        axis_lengths: Dict[int, gtcpp.AxisLength] = field(default_factory=dict)
        _arguments: Set[str] = field(default_factory=set)

        def add_temporaries(
            self, temporaries: List[gtcpp.Temporary]
        ) -> "OIRToGTCpp.GTComputationContext":
            self.temporaries.extend(temporaries)
            return self

        @property
        def arguments(self) -> List[gtcpp.Arg]:
            return [gtcpp.Arg(name=name) for name in self._arguments]

        def add_arguments(self, arguments: Set[str]) -> "OIRToGTCpp.GTComputationContext":
            self._arguments.update(arguments)
            return self

        @staticmethod
        def _make_scalar_accessor(name: str) -> gtcpp.AccessorRef:
            return gtcpp.AccessorRef(
                name=name,
                offset=CartesianOffset.zero(),
                kind=ExprKind.SCALAR,
                dtype=common.DataType.INT32,
            )

        def make_positional(self, axis: int) -> gtcpp.AccessorRef:
            axis_name = ["I", "J", "K"][axis].lower()
            name = self.create_symbol_name(f"ax{axis}_ind")
            positional = self.positionals.setdefault(
                axis, gtcpp.Positional(name=name, axis_name=axis_name)
            )
            return self._make_scalar_accessor(positional.name)

        def make_length(self, axis: int) -> gtcpp.AccessorRef:
            name = self.create_symbol_name(f"ax{axis}_len")
            length = self.axis_lengths.setdefault(axis, gtcpp.AxisLength(name=name, axis=axis))
            return self._make_scalar_accessor(length.name)

        @property
        def extra_decls(self) -> List[gtcpp.ComputationDecl]:
            return list(self.positionals.values()) + list(self.axis_lengths.values())

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
        return gtcpp.Temporary(name=node.name, dtype=node.dtype, data_dims=node.data_dims)

    def visit_VariableKOffset(
        self, node: oir.VariableKOffset, **kwargs: Any
    ) -> gtcpp.VariableKOffset:
        return gtcpp.VariableKOffset(k=self.visit(node.k, **kwargs))

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> gtcpp.AccessorRef:
        return gtcpp.AccessorRef(
            name=node.name,
            offset=self.visit(node.offset, **kwargs),
            data_index=self.visit(node.data_index, **kwargs),
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

    def _mask_to_expr(
        self, mask: common.HorizontalMask, comp_ctx: "GTComputationContext"
    ) -> gtcpp.Expr:
        mask_expr: List[gtcpp.Expr] = []
        for axis_index, interval in enumerate(mask.intervals):
            if interval.is_single_index():
                assert interval.start is not None
                mask_expr.append(
                    gtcpp.BinaryOp(
                        op=common.ComparisonOperator.EQ,
                        left=comp_ctx.make_positional(axis_index),
                        right=_make_axis_offset_expr(
                            interval.start, axis_index, comp_ctx.make_length
                        ),
                    )
                )
            else:
                for op, endpt in zip(
                    (common.ComparisonOperator.GE, common.ComparisonOperator.LT),
                    (interval.start, interval.end),
                ):
                    if endpt is None:
                        continue
                    mask_expr.append(
                        gtcpp.BinaryOp(
                            op=op,
                            left=comp_ctx.make_positional(axis_index),
                            right=_make_axis_offset_expr(endpt, axis_index, comp_ctx.make_length),
                        )
                    )
        return (
            functools.reduce(
                lambda a, b: gtcpp.BinaryOp(op=common.LogicalOperator.AND, left=a, right=b),
                mask_expr,
            )
            if mask_expr
            else gtcpp.Literal(value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL)
        )

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, **kwargs: Any
    ) -> gtcpp.IfStmt:
        mask = self._mask_to_expr(node.mask, kwargs["comp_ctx"])
        return gtcpp.IfStmt(
            cond=mask, true_branch=gtcpp.BlockStmt(body=self.visit(node.body, **kwargs))
        )

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> gtcpp.AssignStmt:
        assert "symtable" in kwargs
        return gtcpp.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> gtcpp.IfStmt:
        return gtcpp.IfStmt(
            cond=self.visit(node.mask, **kwargs),
            true_branch=gtcpp.BlockStmt(body=self.visit(node.body, **kwargs)),
        )

    def visit_While(self, node: oir.While, **kwargs: Any) -> gtcpp.While:
        return gtcpp.While(
            cond=self.visit(node.cond, **kwargs), body=self.visit(node.body, **kwargs)
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        prog_ctx: "ProgramContext",
        comp_ctx: "GTComputationContext",
        interval: gtcpp.GTInterval,
        **kwargs: Any,
    ) -> gtcpp.GTStage:
        assert "symtable" in kwargs
        apply_method = gtcpp.GTApplyMethod(
            interval=self.visit(interval, **kwargs),
            body=self.visit(node.body, comp_ctx=comp_ctx, **kwargs),
            local_variables=self.visit(node.declarations, **kwargs),
        )
        accessors = _extract_accessors(apply_method, {decl.name for decl in comp_ctx.temporaries})
        stage_args = [gtcpp.Arg(name=acc.name) for acc in accessors]

        tmp_names = {tmp.name for tmp in comp_ctx.temporaries}
        param_names_not_tmps = {
            str(param_arg.name) for param_arg in stage_args if param_arg.name not in tmp_names
        }

        comp_ctx.add_arguments(param_names_not_tmps)

        functor_name = type(node).__name__ + str(id(node))
        prog_ctx.add_functor(
            gtcpp.GTFunctor(
                name=functor_name,
                applies=[apply_method],
                param_list=gtcpp.GTParamList(accessors=accessors),
            )
        )

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
        comp_ctx = self.GTComputationContext(
            create_symbol_name=cast(
                SymbolNameCreator, symbol_name_creator(collect_symbol_names(node))
            )
        )

        assert all([isinstance(decl, oir.Temporary) for decl in node.declarations])
        comp_ctx.add_temporaries(self.visit(node.declarations))

        multi_stages = self.visit(
            node.vertical_loops, prog_ctx=prog_ctx, comp_ctx=comp_ctx, **kwargs
        )

        gt_computation = gtcpp.GTComputationCall(
            arguments=comp_ctx.arguments,
            extra_decls=comp_ctx.extra_decls,
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
