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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Union, cast

import dace
import dace.data
import dace.library
import dace.subsets

from gt4py import eve
from gt4py.cartesian.gtc import common, oir
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.expansion_specification import Loop, Map, Sections, Stages
from gt4py.cartesian.gtc.dace.utils import (
    compute_dcir_access_infos,
    flatten_list,
    get_tasklet_symbol,
    make_dace_subset,
    union_inout_memlets,
    union_node_grid_subsets,
    untile_memlets,
)
from gt4py.cartesian.gtc.definitions import Extent

from .utils import remove_horizontal_region


if TYPE_CHECKING:
    from gt4py.cartesian.gtc.dace.nodes import StencilComputation


def _access_iter(node: oir.HorizontalExecution, get_outputs: bool):
    if get_outputs:
        iterator = filter(
            lambda node: isinstance(node, oir.FieldAccess),
            node.walk_values().if_isinstance(oir.AssignStmt).getattr("left"),
        )
    else:

        def _iterator():
            for n in node.walk_values():
                if isinstance(n, oir.AssignStmt):
                    yield from n.right.walk_values().if_isinstance(oir.FieldAccess)
                elif isinstance(n, oir.While):
                    yield from n.cond.walk_values().if_isinstance(oir.FieldAccess)
                elif isinstance(n, oir.MaskStmt):
                    yield from n.mask.walk_values().if_isinstance(oir.FieldAccess)

        iterator = _iterator()

    yield from (
        eve.utils.xiter(iterator).map(
            lambda acc: (
                acc.name,
                acc.offset,
                get_tasklet_symbol(acc.name, offset=acc.offset, is_target=get_outputs),
            )
        )
    ).unique(key=lambda x: x[2])


def _get_tasklet_inout_memlets(
    node: oir.HorizontalExecution,
    *,
    get_outputs: bool,
    global_ctx: DaCeIRBuilder.GlobalContext,
    **kwargs: Any,
) -> List[dcir.Memlet]:
    access_infos = compute_dcir_access_infos(
        node,
        block_extents=global_ctx.library_node.get_extents,
        oir_decls=global_ctx.library_node.declarations,
        collect_read=not get_outputs,
        collect_write=get_outputs,
        **kwargs,
    )

    memlets: List[dcir.Memlet] = []
    for name, offset, tasklet_symbol in _access_iter(node, get_outputs=get_outputs):
        access_info = access_infos[name]
        if not access_info.variable_offset_axes:
            offset_dict = offset.to_dict()
            for axis in access_info.axes():
                access_info = access_info.restricted_to_index(
                    axis, extent=(offset_dict[axis.lower()], offset_dict[axis.lower()])
                )

        memlets.append(
            dcir.Memlet(
                field=name,
                connector=tasklet_symbol,
                access_info=access_info,
                is_read=not get_outputs,
                is_write=get_outputs,
            )
        )
    return memlets


def _all_stmts_same_region(scope_nodes, axis: dcir.Axis, interval: Any) -> bool:
    def all_statements_in_region(scope_nodes: List[eve.Node]) -> bool:
        return all(
            isinstance(stmt, dcir.HorizontalRestriction)
            for tasklet in eve.walk_values(scope_nodes).if_isinstance(dcir.Tasklet)
            for stmt in tasklet.stmts
        )

    def all_regions_same(scope_nodes: List[eve.Node]) -> bool:
        return (
            len(
                set(
                    (
                        (
                            None
                            if mask.intervals[axis.to_idx()].start is None
                            else mask.intervals[axis.to_idx()].start.level
                        ),
                        (
                            None
                            if mask.intervals[axis.to_idx()].start is None
                            else mask.intervals[axis.to_idx()].start.offset
                        ),
                        (
                            None
                            if mask.intervals[axis.to_idx()].end is None
                            else mask.intervals[axis.to_idx()].end.level
                        ),
                        (
                            None
                            if mask.intervals[axis.to_idx()].end is None
                            else mask.intervals[axis.to_idx()].end.offset
                        ),
                    )
                    for mask in eve.walk_values(scope_nodes).if_isinstance(common.HorizontalMask)
                )
            )
            == 1
        )

    return (
        axis in dcir.Axis.dims_horizontal()
        and isinstance(interval, dcir.DomainInterval)
        and all_statements_in_region(scope_nodes)
        and all_regions_same(scope_nodes)
    )


class DaCeIRBuilder(eve.NodeTranslator):
    @dataclass
    class GlobalContext:
        library_node: StencilComputation
        arrays: Dict[str, dace.data.Data]

        def get_dcir_decls(
            self,
            access_infos: Dict[eve.SymbolRef, dcir.FieldAccessInfo],
            symbol_collector: DaCeIRBuilder.SymbolCollector,
        ) -> List[dcir.FieldDecl]:
            return [
                self._get_dcir_decl(field, access_info, symbol_collector=symbol_collector)
                for field, access_info in access_infos.items()
            ]

        def _get_dcir_decl(
            self,
            field: eve.SymbolRef,
            access_info: dcir.FieldAccessInfo,
            symbol_collector: DaCeIRBuilder.SymbolCollector,
        ) -> dcir.FieldDecl:
            oir_decl: oir.Decl = self.library_node.declarations[field]
            assert isinstance(oir_decl, oir.FieldDecl)
            dace_array = self.arrays[field]
            for stride in dace_array.strides:
                for symbol in dace.symbolic.symlist(stride).values():
                    symbol_collector.add_symbol(str(symbol))
            for symbol in access_info.grid_subset.free_symbols:
                symbol_collector.add_symbol(symbol)

            return dcir.FieldDecl(
                name=field,
                dtype=oir_decl.dtype,
                strides=tuple(str(s) for s in dace_array.strides),
                data_dims=oir_decl.data_dims,
                access_info=access_info,
                storage=dcir.StorageType.from_dace_storage(dace.StorageType.Default),
            )

    @dataclass
    class IterationContext:
        grid_subset: dcir.GridSubset
        parent: Optional[DaCeIRBuilder.IterationContext] = None

        def push_axes_extents(self, axes_extents) -> DaCeIRBuilder.IterationContext:
            res = self.grid_subset
            for axis, extent in axes_extents.items():
                axis_interval = res.intervals[axis]
                if isinstance(axis_interval, dcir.DomainInterval):
                    res__interval = dcir.DomainInterval(
                        start=dcir.AxisBound(
                            level=common.LevelMarker.START, offset=extent[0], axis=axis
                        ),
                        end=dcir.AxisBound(
                            level=common.LevelMarker.END, offset=extent[1], axis=axis
                        ),
                    )
                    res = res.set_interval(axis, res__interval)
                elif isinstance(axis_interval, dcir.TileInterval):
                    tile_interval = dcir.TileInterval(
                        axis=axis,
                        start_offset=extent[0],
                        end_offset=extent[1],
                        tile_size=axis_interval.tile_size,
                        domain_limit=axis_interval.domain_limit,
                    )
                    res = res.set_interval(axis, tile_interval)
                # if is IndexWithExtent, do nothing.
            return DaCeIRBuilder.IterationContext(grid_subset=res, parent=self)

        def push_interval(
            self, axis: dcir.Axis, interval: Union[dcir.DomainInterval, oir.Interval]
        ) -> DaCeIRBuilder.IterationContext:
            return DaCeIRBuilder.IterationContext(
                grid_subset=self.grid_subset.set_interval(axis, interval), parent=self
            )

        def push_expansion_item(self, item: Union[Map, Loop]) -> DaCeIRBuilder.IterationContext:
            if not isinstance(item, (Map, Loop)):
                raise ValueError

            iterations = item.iterations if isinstance(item, Map) else [item]
            grid_subset = self.grid_subset
            for it in iterations:
                axis = it.axis
                if it.kind == "tiling":
                    assert it.stride is not None
                    grid_subset = grid_subset.tile(tile_sizes={axis: it.stride})
                else:
                    grid_subset = grid_subset.restricted_to_index(axis)
            return DaCeIRBuilder.IterationContext(grid_subset=grid_subset, parent=self)

        def push_expansion_items(
            self, items: Iterable[Union[Map, Loop]]
        ) -> DaCeIRBuilder.IterationContext:
            res = self
            for item in items:
                res = res.push_expansion_item(item)
            return res

        def pop(self) -> DaCeIRBuilder.IterationContext:
            assert self.parent is not None
            return self.parent

    @dataclass
    class SymbolCollector:
        symbol_decls: Dict[str, dcir.SymbolDecl] = dataclasses.field(default_factory=dict)

        def add_symbol(self, name: str, dtype: common.DataType = common.DataType.INT32) -> None:
            if name not in self.symbol_decls:
                self.symbol_decls[name] = dcir.SymbolDecl(name=name, dtype=dtype)
            else:
                assert self.symbol_decls[name].dtype == dtype

        def remove_symbol(self, name: eve.SymbolRef) -> None:
            if name in self.symbol_decls:
                del self.symbol_decls[name]

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> dcir.Literal:
        return dcir.Literal(value=node.value, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> dcir.UnaryOp:
        return dcir.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs), dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> dcir.BinaryOp:
        return dcir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            dtype=node.dtype,
        )

    def visit_HorizontalRestriction(
        self,
        node: oir.HorizontalRestriction,
        *,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> dcir.HorizontalRestriction:
        for axis, interval in zip(dcir.Axis.dims_horizontal(), node.mask.intervals):
            for bound in (interval.start, interval.end):
                if bound is not None:
                    symbol_collector.add_symbol(axis.iteration_symbol())
                    if bound.level == common.LevelMarker.END:
                        symbol_collector.add_symbol(axis.domain_symbol())

        return dcir.HorizontalRestriction(
            mask=node.mask, body=self.visit(node.body, symbol_collector=symbol_collector, **kwargs)
        )

    def visit_VariableKOffset(
        self, node: oir.VariableKOffset, **kwargs: Any
    ) -> dcir.VariableKOffset:
        return dcir.VariableKOffset(k=self.visit(node.k, **kwargs))

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> dcir.LocalScalarDecl:
        return dcir.LocalScalarDecl(name=node.name, dtype=node.dtype)

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        is_target: bool,
        targets: Set[eve.SymbolRef],
        var_offset_fields: Set[eve.SymbolRef],
        K_write_with_offset: Set[eve.SymbolRef],
        **kwargs: Any,
    ) -> Union[dcir.IndexAccess, dcir.ScalarAccess]:
        """Generate the relevant accessor to match the memlet that was previously setup.

        When a Field is written in K, we force the usage of the OUT memlet throughout the stencil
        to make sure all side effects are being properly resolved. Frontend checks ensure that no
        parallel code issues sips here.
        """

        res: Union[dcir.IndexAccess, dcir.ScalarAccess]
        if node.name in var_offset_fields.union(K_write_with_offset):
            # If write in K, we consider the variable to always be a target
            # is_target = is_target or node.name in targets # or node.name in K_write_with_offset
            name = get_tasklet_symbol(node.name, offset=node.offset, is_target=is_target)
            res = dcir.IndexAccess(
                name=name,
                is_target=is_target,
                offset=self.visit(
                    node.offset,
                    is_target=False,
                    targets=targets,
                    var_offset_fields=var_offset_fields,
                    K_write_with_offset=K_write_with_offset,
                    **kwargs,
                ),
                data_index=node.data_index,
                dtype=node.dtype,
            )
        else:
            is_target = is_target or (
                node.name in targets and node.offset == common.CartesianOffset.zero()
            )
            name = get_tasklet_symbol(node.name, offset=node.offset, is_target=is_target)
            if node.data_index:
                res = dcir.IndexAccess(
                    name=name, offset=None, is_target=is_target, data_index=node.data_index, dtype=node.dtype
                )
            else:
                res = dcir.ScalarAccess(name=name, dtype=node.dtype, is_target=is_target)
        if is_target:
            targets.add(node.name)
        return res

    def visit_ScalarAccess(
        self,
        node: oir.ScalarAccess,
        *,
        is_target: bool,
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> dcir.ScalarAccess:
        if node.name in global_ctx.library_node.declarations:
            # Handle function parameters differently because they are always available
            symbol_collector.add_symbol(node.name, dtype=node.dtype)
            return dcir.ScalarAccess(name=node.name, dtype=node.dtype, is_target=is_target)
        
        # Rename local scalars inside tasklets such that we can pass them from one state
        # to another (same as we do for index access).
        tasklet_name = get_tasklet_symbol(node.name, is_target=is_target)
        return dcir.ScalarAccess(name=tasklet_name, original_name=node.name, dtype=node.dtype, is_target=is_target)

    def visit_AssignStmt(self, node: oir.AssignStmt, *, targets, **kwargs: Any) -> dcir.AssignStmt:
        # Visiting order matters because targets must not contain the target symbols from the left visit
        right = self.visit(node.right, is_target=False, targets=targets, **kwargs)
        left = self.visit(node.left, is_target=True, targets=targets, **kwargs)
        return dcir.AssignStmt(left=left, right=right, loc=node.loc)

    def _condition_tasklet(
        self,
        node: Union[oir.MaskStmt, oir.While],
        *,
        memlets: List[dcir.Memlet],
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any
    ) -> dcir.Tasklet:
        condition_expression = node.mask if isinstance(node, oir.MaskStmt) else node.cond
        tmp_name = f"if_expression_{id(node)}" if isinstance(node, oir.MaskStmt) else f"while_expression_{id(node)}"

        statement = dcir.AssignStmt(
            # Visiting order matters because targets must not contain the target symbols from the left visit
            right=self.visit(condition_expression, is_target=False, global_ctx=global_ctx, symbol_collector=symbol_collector, **kwargs),
            left=dcir.ScalarAccess(name=get_tasklet_symbol(tmp_name, is_target=True), original_name=tmp_name, dtype=common.DataType.BOOL, loc=node.loc, is_target=True),
            loc=node.loc
        )
        read_memlets = []

        # Match node access in this code block with read/write memlets calculated on
        # the full horizontal execution.
        # NOTE We should clean this up in the future.
        for access_node in statement.walk_values().if_isinstance(
            dcir.ScalarAccess, dcir.IndexAccess
        ):
            symbol_name = access_node.name
            matches: List[dcir.Memlet] = [
                memlet for memlet in memlets if memlet.connector == symbol_name
            ]

            if len(matches) > 1:
                raise RuntimeError(
                    "Found more than one matching memlet for symbol '%s'" % symbol_name
                )

            # Memlets aren't hashable, we thus can't use a set
            if (
                len(matches) > 0
                and matches[0].is_write
                and not matches[0].is_read
            ):
                raise RuntimeError("We shouldn't match with write-only memlet")
            if (
                len(matches) > 0
                and matches[0].is_read
                and matches[0] not in read_memlets
            ):
                read_memlets.append(matches[0].copy(update={}))

        tasklet = dcir.Tasklet(
            decls=[],
            stmts=[statement],
            read_memlets=read_memlets,
            write_memlets=[],
        )
        self._fix_memlet_array_access(
            # See notes inside the function
            # NOTE We should clean this up with the node_access matching above
            tasklet=tasklet,
            memlets=read_memlets,
            global_context=global_ctx,
            symbol_collector=symbol_collector,
        )
        tasklet = self._fix_scalar_access_read_after_write(tasklet)
        return tasklet

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        memlets: List[dcir.Memlet],
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> dcir.Condition:
        code_block = oir.CodeBlock(body=node.body, loc=node.loc, label=f"condition_{id(node)}")
        return dcir.Condition(
            condition=self._condition_tasklet(
                node,
                memlets=memlets,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
                **kwargs,
            ),
            true_state=self.visit(code_block, memlets=memlets, global_ctx=global_ctx, symbol_collector=symbol_collector, **kwargs),
        )

    def visit_While(
        self,
        node: oir.While,
        memlets: List[dcir.Memlet],
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> dcir.WhileLoop:
        code_block = oir.CodeBlock(body=node.body, loc=node.loc, label=f"while_{id(node)}")
        return dcir.WhileLoop(
            condition=self._condition_tasklet(
                node,
                memlets=memlets,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
                **kwargs,
            ),
            body=self.visit(code_block, memlets=memlets, global_ctx=global_ctx, symbol_collector=symbol_collector, **kwargs),
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> dcir.Cast:
        return dcir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> dcir.NativeFuncCall:
        return dcir.NativeFuncCall(
            func=node.func, args=self.visit(node.args, **kwargs), dtype=node.dtype
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> dcir.TernaryOp:
        return dcir.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
            dtype=node.dtype,
        )
    
    def _fix_scalar_access_read_after_write(self, tasklet: dcir.Tasklet) -> dcir.Tasklet:
        """
        This function fixes read after write situations for scalar accesses within a Tasklet.

        We can't detect these (in general) when building the DaCe IR directly because we don't
        know where the "Tasklet boundaries" are.
        """

        class FixScalarReadAfterWrite(eve.NodeTranslator):
            def visit_ScalarAccess(self, node: dcir.ScalarAccess, targets: Set[eve.SymbolRef], **kwargs: Any) -> dcir.ScalarAccess:
                if node.original_name is None:
                    # Field access can be represented as ScalarAccess. They don't have an original_name
                    # and that is fine. We shouldn't mess with them.
                    return copy.deepcopy(node)
                
                # Handle write
                if node.is_target:
                    targets.add(node.original_name)
                    return copy.deepcopy(node)
                
                # Handle read after write
                if node.original_name in targets:
                    new_name = get_tasklet_symbol(node.original_name, is_target=True)
                    return dcir.ScalarAccess(name=new_name, original_name=node.original_name, loc=node.loc, dtype=node.dtype, is_target=node.is_target)
                
                # Handle general read
                return copy.deepcopy(node)
                

            def visit_AssignStmt(self, node: dcir.AssignStmt, **kwargs: Any) -> dcir.AssignStmt:
                return dcir.AssignStmt(
                    # Visiting order matters because targets must not contain the target symbols from the left visit
                    right=self.visit(node.right, **kwargs),
                    left=self.visit(node.left, **kwargs),
                    loc=node.loc,
                )

            #@classmethod
            #def apply_fix(cls, node: dcir.Tasklet, **kwargs: Any) -> dcir.Tasklet:
            #    # NOTE This is not named 'apply' b/c the base class has a method with
            #    # that name and a different type signature.
            #    if not isinstance(node, dcir.Tasklet):
            #        raise ValueError("apply() requires dcir.Tasklet node")
            #    
            #    
            #    return super().apply(node, targets=targets, **kwargs)

        # Scalars that we write to inside this Tasklet.
        targets: Set[eve.SymbolRef] = set()
        return FixScalarReadAfterWrite().visit(tasklet, targets=targets)

        ## Scalars that we write to inside this Tasklet.
        #targets: Set[eve.SymbolRef] = set()
        #
        #for node in tasklet.walk_values().if_isinstance(dcir.ScalarAccess):
        #    # Our type hint system isn't good enough. Even though we filter on Scalar_Access
        #    # nodes, `node` is typed as `Any`.
        #    assert isinstance(node, dcir.ScalarAccess)
        #
        #    # Field access can be represented as ScalarAccess. They don't have an original_name
        #    # and that is fine. We shouldn't mess with them.
        #    if node.original_name is None:
        #        continue
        #
        #    # Handle write access
        #    if node.is_target:
        #        targets.add(node.original_name)
        #        continue
        #
        #    # Handle read access
        #    if node.original_name in targets:
        #        # Read after write detected
        #        node.name = get_tasklet_symbol(node.original_name, is_target=True)

    def _fix_memlet_array_access(
        self,
        *,
        tasklet: dcir.Tasklet,
        memlets: List[dcir.Memlet],
        global_context: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
    ) -> None:
        for memlet in memlets:
            """
            This loop handles the special case of a tasklet performing array access.
            The memlet should pass the full array shape (no tiling) and
            the tasklet expression for array access should use all explicit indexes.
            """
            array_ndims = len(global_context.arrays[memlet.field].shape)
            field_decl = global_context.library_node.field_decls[memlet.field]
            # calculate array subset on original memlet
            memlet_subset = make_dace_subset(
                global_context.library_node.access_infos[memlet.field],
                memlet.access_info,
                field_decl.data_dims,
            )
            # select index values for single-point grid access
            memlet_data_index = [
                dcir.Literal(value=str(dim_range[0]), dtype=common.DataType.INT32)
                for dim_range, dim_size in zip(memlet_subset, memlet_subset.size())
                if dim_size == 1
            ]
            if len(memlet_data_index) < array_ndims:
                reshape_memlet = False
                for access_node in tasklet.walk_values().if_isinstance(dcir.IndexAccess):
                    if access_node.data_index and access_node.name == memlet.connector:
                        access_node.data_index = memlet_data_index + access_node.data_index
                        assert len(access_node.data_index) == array_ndims
                        reshape_memlet = True
                if reshape_memlet:
                    # ensure that memlet symbols used for array indexing are defined in context
                    for sym in memlet.access_info.grid_subset.free_symbols:
                        symbol_collector.add_symbol(sym)
                    # set full shape on memlet
                    memlet.access_info = global_context.library_node.access_infos[memlet.field]

    def visit_CodeBlock(
        self,
        node: oir.CodeBlock,
        *,
        global_ctx: DaCeIRBuilder.GlobalContext,
        iteration_ctx: DaCeIRBuilder.IterationContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        memlets: Optional[List[dcir.Memlet]] = None,
        k_interval,
        **kwargs: Any,
    ):
        memlets = [] if memlets is None else memlets
        targets: Set[str] = set()
        kwargs.pop("targets", "dummy")
        statements = [
            self.visit(
                statement,
                targets=targets,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
                iteration_ctx=iteration_ctx,
                k_interval=k_interval,
                memlets=memlets,
                **kwargs,
            )
            for statement in node.body
        ]

        # gather all statements that aren't control flow (e.g. Condition or WhileLoop)
        # put them in a tasklet
        # and call "to_state" on it
        # then, return new list with types that are either ComputationState, Condition, or WhileLoop
        dace_nodes: List[Union[dcir.ComputationState, dcir.Condition, dcir.WhileLoop]] = []
        current_block: List[dcir.Stmt] = []
        for index, statement in enumerate(statements):
            is_control_flow = isinstance(statement, dcir.Condition) or isinstance(
                statement, dcir.WhileLoop
            )
            if not is_control_flow:
                current_block.append(statement)

            last_statement = index == len(statements) - 1
            if (is_control_flow or last_statement) and len(current_block) > 0:
                read_memlets: List[dcir.Memlet] = []
                write_memlets: List[dcir.Memlet] = []

                for block_statement in current_block:
                    # Match node access in this code block with read/write memlets calculated on
                    # the full horizontal execution.
                    # NOTE We should clean this up in the future.
                    for access_node in block_statement.walk_values().if_isinstance(
                        dcir.ScalarAccess, dcir.IndexAccess
                    ):
                        symbol_name = access_node.name
                        matches: List[dcir.Memlet] = [
                            memlet for memlet in memlets if memlet.connector == symbol_name
                        ]

                        if len(matches) > 1:
                            raise RuntimeError(
                                "Found more than one matching memlet for symbol '%s'" % symbol_name
                            )

                        # Memlets aren't hashable, we thus can't use a set
                        if (
                            len(matches) > 0
                            and matches[0].is_write
                            and matches[0] not in write_memlets
                        ):
                            write_memlets.append(matches[0].copy(update={}))
                        if (
                            len(matches) > 0
                            and matches[0].is_read
                            and matches[0] not in read_memlets
                        ):
                            read_memlets.append(matches[0].copy(update={}))

                # create a new tasklet
                tasklet = dcir.Tasklet(
                    decls=[],
                    stmts=current_block,
                    read_memlets=read_memlets,
                    write_memlets=write_memlets,
                )
                self._fix_memlet_array_access(
                    # See notes inside the function
                    # NOTE We should clean this up with the node_access matching above
                    tasklet=tasklet,
                    memlets=[*read_memlets, *write_memlets],
                    global_context=global_ctx,
                    symbol_collector=symbol_collector,
                )
                tasklet = self._fix_scalar_access_read_after_write(tasklet)

                dace_nodes.append(*self.to_state(tasklet, grid_subset=iteration_ctx.grid_subset))

                # reset block scope
                current_block = []

            # append control flow statement after new tasklet (if applicable)
            if is_control_flow:
                dace_nodes.append(statement)

        return dace_nodes

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        global_ctx: DaCeIRBuilder.GlobalContext,
        iteration_ctx: DaCeIRBuilder.IterationContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        k_interval,
        **kwargs: Any,
    ):
        extent = global_ctx.library_node.get_extents(node)

        stages_idx = next(
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, Stages)
        )
        expansion_items = global_ctx.library_node.expansion_specification[stages_idx + 1 :]

        iteration_ctx = iteration_ctx.push_axes_extents(
            {k: v for k, v in zip(dcir.Axis.dims_horizontal(), extent)}
        )
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)
        assert iteration_ctx.grid_subset == dcir.GridSubset.single_gridpoint()

        read_memlets = _get_tasklet_inout_memlets(
            node,
            get_outputs=False,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        write_memlets = _get_tasklet_inout_memlets(
            node,
            get_outputs=True,
            global_ctx=global_ctx,
            grid_subset=iteration_ctx.grid_subset,
            k_interval=k_interval,
        )

        # local_scalar_declarations: List[dcir.LocalScalarDecl] = [
        #     self.visit(declaration, **kwargs) for declaration in node.declarations
        # ]

        code_block = oir.CodeBlock(body=node.body, loc=node.loc, label=f"he_{id(node)}")

        dcir_nodes = self.visit(
            code_block,
            global_ctx=global_ctx,
            iteration_ctx=iteration_ctx,
            symbol_collector=symbol_collector,
            k_interval=k_interval,
            memlets=[*read_memlets, *write_memlets],
            # declarations=local_scalar_declarations,
            **kwargs,
        )

        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            dcir_nodes = self._process_iteration_item(
                dcir_nodes,
                item,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                symbol_collector=symbol_collector,
                **kwargs,
            )
        # pop stages context (pushed with push_grid_subset)
        iteration_ctx.pop()

        return dcir_nodes

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        iteration_ctx: DaCeIRBuilder.IterationContext,
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs,
    ):
        sections_idx, stages_idx = [
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, (Sections, Stages))
        ]
        expansion_items = global_ctx.library_node.expansion_specification[
            sections_idx + 1 : stages_idx
        ]

        iteration_ctx = iteration_ctx.push_interval(
            dcir.Axis.K, node.interval
        ).push_expansion_items(expansion_items)

        dcir_nodes = self.generic_visit(
            node.horizontal_executions,
            iteration_ctx=iteration_ctx,
            global_ctx=global_ctx,
            symbol_collector=symbol_collector,
            k_interval=node.interval,
            **kwargs,
        )

        # if multiple horizontal executions, enforce their order by means of a state machine
        if len(dcir_nodes) > 1:
            dcir_nodes = [
                self.to_state([node], grid_subset=node.grid_subset)
                for node in flatten_list(dcir_nodes)
            ]

        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            dcir_nodes = self._process_iteration_item(
                scope=dcir_nodes,
                item=item,
                iteration_ctx=iteration_ctx,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
            )
        # pop off interval
        iteration_ctx.pop()
        return dcir_nodes

    def to_dataflow(
        self,
        nodes,
        *,
        global_ctx: DaCeIRBuilder.GlobalContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
    ):
        nodes = flatten_list(nodes)
        if all(isinstance(n, (dcir.NestedSDFG, dcir.DomainMap, dcir.Tasklet)) for n in nodes):
            return nodes
        if not all(isinstance(n, (dcir.ComputationState, dcir.Condition, dcir.DomainLoop, dcir.WhileLoop)) for n in nodes):
            raise ValueError("Can't mix dataflow and state nodes on same level.")

        read_memlets, write_memlets, field_memlets = union_inout_memlets(nodes)

        field_decls = global_ctx.get_dcir_decls(
            {memlet.field: memlet.access_info for memlet in field_memlets},
            symbol_collector=symbol_collector,
        )
        read_fields = {memlet.field for memlet in read_memlets}
        write_fields = {memlet.field for memlet in write_memlets}
        read_memlets = [
            memlet.remove_write() for memlet in field_memlets if memlet.field in read_fields
        ]
        write_memlets = [
            memlet.remove_read() for memlet in field_memlets if memlet.field in write_fields
        ]

        return [
            dcir.NestedSDFG(
                label=global_ctx.library_node.label,
                field_decls=field_decls,
                # NestedSDFG must have same shape on input and output, matching corresponding
                # nsdfg.sdfg's array shape
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                states=nodes,
                symbol_decls=list(symbol_collector.symbol_decls.values()),
            )
        ]

    def to_state(self, nodes, *, grid_subset: dcir.GridSubset):
        nodes = flatten_list(nodes)
        if all(
            isinstance(n, (dcir.ComputationState, dcir.Condition, dcir.DomainLoop, dcir.WhileLoop))
            for n in nodes
        ):
            return nodes
        if all(isinstance(n, (dcir.DomainMap, dcir.NestedSDFG, dcir.Tasklet)) for n in nodes):
            return [dcir.ComputationState(computations=nodes, grid_subset=grid_subset)]

        raise ValueError("Can't mix dataflow and state nodes on same level.")

    def _process_map_item(
        self,
        scope_nodes,
        item: Map,
        *,
        global_ctx: DaCeIRBuilder.GlobalContext,
        iteration_ctx: DaCeIRBuilder.IterationContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> List[dcir.DomainMap]:
        grid_subset = iteration_ctx.grid_subset
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_dataflow(
            scope_nodes, global_ctx=global_ctx, symbol_collector=symbol_collector
        )

        ranges = []
        for iteration in item.iterations:
            axis = iteration.axis
            interval = iteration_ctx.grid_subset.intervals[axis]
            grid_subset = grid_subset.set_interval(axis, interval)
            if iteration.kind == "tiling":
                read_memlets = untile_memlets(read_memlets, axes=[axis])
                write_memlets = untile_memlets(write_memlets, axes=[axis])
                if not axis == dcir.Axis.K:
                    interval = dcir.DomainInterval(
                        start=dcir.AxisBound.from_common(axis, oir.AxisBound.start()),
                        end=dcir.AxisBound.from_common(axis, oir.AxisBound.end()),
                    )
                symbol_collector.remove_symbol(axis.tile_symbol())
                ranges.append(
                    dcir.Range(var=axis.tile_symbol(), interval=interval, stride=iteration.stride)
                )
            else:
                if _all_stmts_same_region(scope_nodes, axis, interval):
                    masks = cast(
                        List[common.HorizontalMask],
                        eve.walk_values(scope_nodes).if_isinstance(common.HorizontalMask).to_list(),
                    )
                    horizontal_mask_interval = next(
                        iter((mask.intervals[axis.to_idx()] for mask in masks))
                    )
                    interval = dcir.DomainInterval.intersection(
                        axis, horizontal_mask_interval, interval
                    )
                    scope_nodes = remove_horizontal_region(scope_nodes, axis)
                assert iteration.kind == "contiguous"
                res_read_memlets = []
                res_write_memlets = []
                for memlet in read_memlets:
                    access_info = memlet.access_info.apply_iteration(
                        dcir.GridSubset.from_interval(interval, axis)
                    )
                    for sym in access_info.grid_subset.free_symbols:
                        symbol_collector.add_symbol(sym)
                    res_read_memlets.append(
                        dcir.Memlet(
                            field=memlet.field,
                            connector=memlet.connector,
                            access_info=access_info,
                            is_read=True,
                            is_write=False,
                        )
                    )
                for memlet in write_memlets:
                    access_info = memlet.access_info.apply_iteration(
                        dcir.GridSubset.from_interval(interval, axis)
                    )
                    for sym in access_info.grid_subset.free_symbols:
                        symbol_collector.add_symbol(sym)
                    res_write_memlets.append(
                        dcir.Memlet(
                            field=memlet.field,
                            connector=memlet.connector,
                            access_info=access_info,
                            is_read=False,
                            is_write=True,
                        )
                    )
                read_memlets = res_read_memlets
                write_memlets = res_write_memlets

                assert not isinstance(interval, dcir.IndexWithExtent)
                index_range = dcir.Range.from_axis_and_interval(axis, interval)
                symbol_collector.remove_symbol(index_range.var)
                ranges.append(index_range)

        return [
            dcir.DomainMap(
                computations=scope_nodes,
                index_ranges=ranges,
                schedule=dcir.MapSchedule.from_dace_schedule(item.schedule),
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                grid_subset=grid_subset,
            )
        ]

    def _process_loop_item(
        self,
        scope_nodes,
        item: Loop,
        *,
        iteration_ctx: DaCeIRBuilder.IterationContext,
        symbol_collector: DaCeIRBuilder.SymbolCollector,
        **kwargs: Any,
    ) -> List[dcir.DomainLoop]:
        grid_subset = union_node_grid_subsets(list(scope_nodes))
        read_memlets, write_memlets, _ = union_inout_memlets(list(scope_nodes))
        scope_nodes = self.to_state(scope_nodes, grid_subset=grid_subset)

        axis = item.axis
        interval = iteration_ctx.grid_subset.intervals[axis]
        grid_subset = grid_subset.set_interval(axis, interval)
        if item.kind == "tiling":
            raise NotImplementedError("Tiling as a state machine not implemented.")

        assert item.kind == "contiguous"
        res_read_memlets = []
        res_write_memlets = []
        for memlet in read_memlets:
            access_info = memlet.access_info.apply_iteration(
                dcir.GridSubset.from_interval(interval, axis)
            )
            for sym in access_info.grid_subset.free_symbols:
                symbol_collector.add_symbol(sym)
            res_read_memlets.append(
                dcir.Memlet(
                    field=memlet.field,
                    connector=memlet.connector,
                    access_info=access_info,
                    is_read=True,
                    is_write=False,
                )
            )
        for memlet in write_memlets:
            access_info = memlet.access_info.apply_iteration(
                dcir.GridSubset.from_interval(interval, axis)
            )
            for sym in access_info.grid_subset.free_symbols:
                symbol_collector.add_symbol(sym)
            res_write_memlets.append(
                dcir.Memlet(
                    field=memlet.field,
                    connector=memlet.connector,
                    access_info=access_info,
                    is_read=False,
                    is_write=True,
                )
            )
        read_memlets = res_read_memlets
        write_memlets = res_write_memlets

        assert not isinstance(interval, dcir.IndexWithExtent)
        index_range = dcir.Range.from_axis_and_interval(axis, interval, stride=item.stride)
        for sym in index_range.free_symbols:
            symbol_collector.add_symbol(sym, common.DataType.INT32)
        symbol_collector.remove_symbol(index_range.var)
        return [
            dcir.DomainLoop(
                axis=axis,
                loop_states=scope_nodes,
                index_range=index_range,
                read_memlets=read_memlets,
                write_memlets=write_memlets,
                grid_subset=grid_subset,
            )
        ]

    def _process_iteration_item(self, scope, item, **kwargs):
        if isinstance(item, Map):
            return self._process_map_item(scope, item, **kwargs)
        if isinstance(item, Loop):
            return self._process_loop_item(scope, item, **kwargs)

        raise ValueError("Invalid expansion specification set.")

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, global_ctx: DaCeIRBuilder.GlobalContext, **kwargs: Any
    ) -> dcir.NestedSDFG:
        overall_extent = Extent.zeros(2)
        for he in node.walk_values().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(global_ctx.library_node.get_extents(he))

        iteration_ctx = DaCeIRBuilder.IterationContext(
            grid_subset=dcir.GridSubset.from_gt4py_extent(overall_extent).set_interval(
                axis=dcir.Axis.K, interval=node.sections[0].interval
            )
        )

        # Variable offsets
        var_offset_fields = {
            acc.name
            for acc in node.walk_values().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, oir.VariableKOffset)
        }

        # We book keep - all write offset to K
        K_write_with_offset = set()
        for assign_node in node.walk_values().if_isinstance(oir.AssignStmt):
            if isinstance(assign_node.left, oir.FieldAccess):
                if (
                    isinstance(assign_node.left.offset, common.CartesianOffset)
                    and assign_node.left.offset.k != 0
                ):
                    K_write_with_offset.add(assign_node.left.name)

        sections_idx = next(
            idx
            for idx, item in enumerate(global_ctx.library_node.expansion_specification)
            if isinstance(item, Sections)
        )
        expansion_items = global_ctx.library_node.expansion_specification[:sections_idx]
        iteration_ctx = iteration_ctx.push_expansion_items(expansion_items)

        symbol_collector = DaCeIRBuilder.SymbolCollector()
        sections = flatten_list(
            self.generic_visit(
                node.sections,
                global_ctx=global_ctx,
                iteration_ctx=iteration_ctx,
                symbol_collector=symbol_collector,
                var_offset_fields=var_offset_fields,
                K_write_with_offset=K_write_with_offset,
                **kwargs,
            )
        )
        if node.loop_order != common.LoopOrder.PARALLEL:
            sections = [self.to_state(s, grid_subset=iteration_ctx.grid_subset) for s in sections]
        computations = sections
        for item in reversed(expansion_items):
            iteration_ctx = iteration_ctx.pop()
            computations = self._process_iteration_item(
                scope=computations,
                item=item,
                iteration_ctx=iteration_ctx,
                global_ctx=global_ctx,
                symbol_collector=symbol_collector,
            )

        read_memlets, write_memlets, field_memlets = union_inout_memlets(computations)

        field_decls = global_ctx.get_dcir_decls(
            global_ctx.library_node.access_infos, symbol_collector=symbol_collector
        )

        read_fields = set(memlet.field for memlet in read_memlets)
        write_fields = set(memlet.field for memlet in write_memlets)
        return dcir.NestedSDFG(
            label=global_ctx.library_node.label,
            states=self.to_state(computations, grid_subset=iteration_ctx.grid_subset),
            field_decls=field_decls,
            read_memlets=[memlet for memlet in field_memlets if memlet.field in read_fields],
            write_memlets=[memlet for memlet in field_memlets if memlet.field in write_fields],
            symbol_decls=list(symbol_collector.symbol_decls.values()),
        )
