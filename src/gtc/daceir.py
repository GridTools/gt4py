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

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import dace
import sympy

import eve
import gtc
import gtc.definitions
from eve import datamodels
from gtc import common, oir
from gtc.common import LocNode
from gtc.dace.symbol_utils import (
    get_axis_bound_dace_symbol,
    get_axis_bound_diff_str,
    get_axis_bound_str,
    get_dace_symbol,
)


@eve.utils.noninstantiable
class Expr(common.Expr):
    dtype: common.DataType


@eve.utils.noninstantiable
class Stmt(common.Stmt):
    pass


class Axis(eve.StrEnum):
    I = "I"  # noqa: E741 ambiguous variable name 'I'
    J = "J"
    K = "K"

    def domain_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__" + self.upper())

    def iteration_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__" + self.lower())

    def tile_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef("__tile_" + self.lower())

    @staticmethod
    def dims_3d() -> Generator["Axis", None, None]:
        yield from [Axis.I, Axis.J, Axis.K]

    @staticmethod
    def dims_horizontal() -> Generator["Axis", None, None]:
        yield from [Axis.I, Axis.J]

    def to_idx(self) -> int:
        return [Axis.I, Axis.J, Axis.K].index(self)

    def domain_dace_symbol(self):
        return get_dace_symbol(self.domain_symbol())

    def iteration_dace_symbol(self):
        return get_dace_symbol(self.iteration_symbol())

    def tile_dace_symbol(self):
        return get_dace_symbol(self.tile_symbol())


class MapSchedule(eve.IntEnum):
    Default = 0
    Sequential = 1

    CPU_Multicore = 2

    GPU_Device = 3
    GPU_ThreadBlock = 4

    def to_dace_schedule(self):
        return {
            MapSchedule.Default: dace.ScheduleType.Default,
            MapSchedule.Sequential: dace.ScheduleType.Sequential,
            MapSchedule.CPU_Multicore: dace.ScheduleType.CPU_Multicore,
            MapSchedule.GPU_Device: dace.ScheduleType.GPU_Device,
            MapSchedule.GPU_ThreadBlock: dace.ScheduleType.GPU_ThreadBlock,
        }[self]

    @classmethod
    def from_dace_schedule(cls, schedule):
        return {
            dace.ScheduleType.Default: MapSchedule.Default,
            dace.ScheduleType.Sequential: MapSchedule.Sequential,
            dace.ScheduleType.CPU_Multicore: MapSchedule.CPU_Multicore,
            dace.ScheduleType.GPU_Device: MapSchedule.GPU_Device,
            dace.ScheduleType.GPU_ThreadBlock: MapSchedule.GPU_ThreadBlock,
        }[schedule]


class StorageType(eve.IntEnum):
    Default = 0

    CPU_Heap = 1

    GPU_Global = 3
    GPU_Shared = 4

    Register = 5

    def to_dace_storage(self):
        return {
            StorageType.Default: dace.StorageType.Default,
            StorageType.CPU_Heap: dace.StorageType.CPU_Heap,
            StorageType.GPU_Global: dace.StorageType.GPU_Global,
            StorageType.GPU_Shared: dace.StorageType.GPU_Shared,
            StorageType.Register: dace.StorageType.Register,
        }[self]

    @classmethod
    def from_dace_storage(cls, schedule):
        return {
            dace.StorageType.Default: StorageType.Default,
            dace.StorageType.CPU_Heap: StorageType.CPU_Heap,
            dace.StorageType.GPU_Global: StorageType.GPU_Global,
            dace.StorageType.GPU_Shared: StorageType.GPU_Shared,
            dace.StorageType.Register: StorageType.Register,
        }[schedule]


class AxisBound(common.AxisBound):
    axis: Axis

    def __str__(self):
        return get_axis_bound_str(self, self.axis.domain_symbol())

    @classmethod
    def from_common(cls, axis, node):
        return cls(axis=axis, level=node.level, offset=node.offset)

    def to_dace_symbolic(self):
        return get_axis_bound_dace_symbol(self)


class IndexWithExtent(eve.Node):
    axis: Axis
    value: Union[AxisBound, int, str]
    extent: Tuple[int, int]

    @property
    def free_symbols(self) -> Set[eve.SymbolRef]:
        if isinstance(self.value, AxisBound) and self.value.level == common.LevelMarker.END:
            return {self.axis.domain_symbol()}
        elif isinstance(self.value, str):
            return {self.axis.iteration_symbol()}
        return set()

    @classmethod
    def from_axis(cls, axis: Axis, extent=(0, 0)):
        return cls(axis=axis, value=axis.iteration_symbol(), extent=extent)

    @property
    def size(self):
        return self.extent[1] - self.extent[0] + 1

    @property
    def overapproximated_size(self):
        return self.size

    def union(self, other: "IndexWithExtent"):
        assert self.axis == other.axis
        if isinstance(self.value, int) or (isinstance(self.value, str) and self.value.isdigit()):
            value = other.value
        elif isinstance(other.value, int) or (
            isinstance(other.value, str) and other.value.isdigit()
        ):
            value = self.value
        elif (
            self.value == self.axis.iteration_symbol()
            or other.value == self.axis.iteration_symbol()
        ):
            value = self.axis.iteration_symbol()
        else:
            assert other.value == self.value
            value = self.value
        return IndexWithExtent(
            axis=self.axis,
            value=value,
            extent=(
                min(self.extent[0], other.extent[0]),
                max(self.extent[1], other.extent[1]),
            ),
        )

    @property
    def idx_range(self):
        return (
            f"{self.value}{self.extent[0]:+d}",
            f"{self.value}{self.extent[1] + 1:+d}",
        )

    def to_dace_symbolic(self):
        if isinstance(self.value, AxisBound):
            symbolic_value = get_axis_bound_dace_symbol(self.value)
        elif isinstance(self.value, str):
            symbolic_value = next(
                axis for axis in Axis.dims_3d() if axis.iteration_symbol() == self.value
            ).iteration_dace_symbol()
        else:
            symbolic_value = self.value
        return symbolic_value + self.extent[0], symbolic_value + self.extent[1] + 1

    def shifted(self, offset):
        extent = self.extent[0] + offset, self.extent[1] + offset
        return IndexWithExtent(axis=self.axis, value=self.value, extent=extent)


class DomainInterval(eve.Node):
    start: AxisBound
    end: AxisBound

    @property
    def free_symbols(self) -> Set[eve.SymbolRef]:
        res = set()
        if self.start.level == common.LevelMarker.END:
            res.add(self.start.axis.domain_symbol())
        if self.end.level == common.LevelMarker.END:
            res.add(self.end.axis.domain_symbol())
        return res

    @property
    def size(self):
        return get_axis_bound_diff_str(
            self.end, self.start, var_name=self.start.axis.domain_symbol()
        )

    @property
    def overapproximated_size(self):
        return self.size

    @classmethod
    def union(cls, first, second):
        return cls(
            start=min(first.start, second.start),
            end=max(first.end, second.end),
        )

    @classmethod
    def intersection(cls, axis, first, second):
        first_start = first.start if first.start is not None else second.start
        first_end = first.end if first.end is not None else second.end
        second_start = second.start if second.start is not None else first.start
        second_end = second.end if second.end is not None else first_end.end

        assert (first_start <= second_end and first_end >= second_start) or (
            second_start <= first_end and second_end >= first_start
        )

        start = max(first_start, second_start)
        start = AxisBound(axis=axis, level=start.level, offset=start.offset)
        end = min(first_end, second_end)
        end = AxisBound(axis=axis, level=end.level, offset=end.offset)
        return cls(start=start, end=end)

    @property
    def idx_range(self):
        return str(self.start), str(self.end)

    def to_dace_symbolic(self):
        return self.start.to_dace_symbolic(), self.end.to_dace_symbolic()

    def shifted(self, offset: int):
        return DomainInterval(
            start=AxisBound(
                axis=self.start.axis,
                level=self.start.level,
                offset=self.start.offset + offset,
            ),
            end=AxisBound(
                axis=self.end.axis,
                level=self.end.level,
                offset=self.end.offset + offset,
            ),
        )

    def is_subset_of(self, other: "DomainInterval") -> bool:
        return self.start >= other.start and self.end <= other.end


class TileInterval(eve.Node):
    axis: Axis
    start_offset: int
    end_offset: int
    tile_size: int
    domain_limit: AxisBound

    @property
    def free_symbols(self) -> Set[eve.SymbolRef]:
        res = {
            self.axis.tile_symbol(),
        }
        if self.domain_limit.level == common.LevelMarker.END:
            res.add(self.axis.domain_symbol())
        return res

    @property
    def size(self):
        return "min({tile_size}, {domain_limit} - {tile_symbol}){halo_size:+d}".format(
            tile_size=self.tile_size,
            domain_limit=self.domain_limit,
            tile_symbol=self.axis.tile_symbol(),
            halo_size=self.end_offset - self.start_offset,
        )

    @property
    def overapproximated_size(self):
        return "{tile_size}{halo_size:+d}".format(
            tile_size=self.tile_size,
            halo_size=self.end_offset - self.start_offset,
        )

    @classmethod
    def union(cls, first, second):
        assert first.axis == second.axis
        assert first.tile_size == second.tile_size
        assert first.domain_limit == second.domain_limit
        return cls(
            axis=first.axis,
            start_offset=min(first.start_offset, second.start_offset),
            end_offset=max(first.end_offset, second.end_offset),
            tile_size=first.tile_size,
            domain_limit=first.domain_limit,
        )

    @property
    def idx_range(self):
        start = f"{self.axis.tile_symbol()}{self.start_offset:+d}"
        end = f"{start}+({self.size})"
        return start, end

    def dace_symbolic_size(self):
        return (
            sympy.Min(
                self.tile_size, self.domain_limit.to_dace_symbolic() - self.axis.tile_dace_symbol()
            )
            + self.end_offset
            - self.start_offset
        )

    def to_dace_symbolic(self):

        start = self.axis.tile_dace_symbol() + self.start_offset
        end = start + self.dace_symbolic_size()
        return start, end


class Range(eve.Node):
    var: eve.SymbolRef
    interval: Union[DomainInterval, TileInterval]
    stride: int

    @classmethod
    def from_axis_and_interval(
        cls, axis: Axis, interval: Union[DomainInterval, TileInterval], stride=1
    ):

        return cls(
            var=axis.iteration_symbol(),
            interval=interval,
            stride=stride,
        )

    @property
    def free_symbols(self) -> Set[eve.SymbolRef]:
        return {self.var, *self.interval.free_symbols}


class GridSubset(eve.Node):
    intervals: Dict[Axis, Union[DomainInterval, TileInterval, IndexWithExtent]]

    def __iter__(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield self.intervals[axis]

    def items(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield axis, self.intervals[axis]

    @property
    def free_symbols(self) -> Set[eve.SymbolRef]:
        return set().union(*(interval.free_symbols for interval in self.intervals.values()))

    @classmethod
    def single_gridpoint(cls, offset=(0, 0, 0)):
        return cls(
            intervals={
                axis: IndexWithExtent.from_axis(axis, extent=(offset[i], offset[i]))
                for i, axis in enumerate(Axis.dims_3d())
            }
        )

    @property
    def shape(self):
        return tuple(interval.size for _, interval in self.items())

    @property
    def overapproximated_shape(self):
        return tuple(interval.overapproximated_size for _, interval in self.items())

    def restricted_to_index(self, axis: Axis, extent=(0, 0)) -> "GridSubset":
        intervals = dict(self.intervals)
        intervals[axis] = IndexWithExtent.from_axis(axis, extent=extent)
        return GridSubset(intervals=intervals)

    def set_interval(
        self,
        axis: Axis,
        interval: Union[DomainInterval, IndexWithExtent, TileInterval, oir.Interval],
    ) -> "GridSubset":
        if isinstance(interval, oir.Interval):
            interval = DomainInterval(
                start=AxisBound(
                    level=interval.start.level,
                    offset=interval.start.offset,
                    axis=Axis.K,
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        elif isinstance(interval, DomainInterval):
            assert interval.start.axis == axis
        intervals = dict(self.intervals)
        intervals[axis] = interval
        return GridSubset(intervals=intervals)

    @classmethod
    def from_gt4py_extent(cls, extent: gtc.definitions.Extent):
        i_interval = DomainInterval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=Axis.I),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=Axis.I),
        )
        j_interval = DomainInterval(
            start=AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=Axis.J),
            end=AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=Axis.J),
        )

        return cls(intervals={Axis.I: i_interval, Axis.J: j_interval})

    @classmethod
    def from_interval(
        cls,
        interval: Union[oir.Interval, TileInterval, DomainInterval, IndexWithExtent],
        axis: Axis,
    ):
        res_interval: Union[IndexWithExtent, TileInterval, DomainInterval]
        if isinstance(interval, (DomainInterval, oir.Interval)):
            res_interval = DomainInterval(
                start=AxisBound(
                    level=interval.start.level,
                    offset=interval.start.offset,
                    axis=Axis.K,
                ),
                end=AxisBound(level=interval.end.level, offset=interval.end.offset, axis=Axis.K),
            )
        else:
            assert isinstance(interval, (TileInterval, IndexWithExtent))
            res_interval = interval

        return cls(intervals={axis: res_interval})

    def axes(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield axis

    @classmethod
    def full_domain(cls, axes=None):
        if axes is None:
            axes = Axis.dims_3d()
        res_subsets = dict()
        for axis in axes:
            res_subsets[axis] = DomainInterval(
                start=AxisBound(axis=axis, level=common.LevelMarker.START, offset=0),
                end=AxisBound(axis=axis, level=common.LevelMarker.END, offset=0),
            )
        return GridSubset(intervals=res_subsets)

    def tile(self, tile_sizes: Dict[Axis, int]):
        res_intervals: Dict[Axis, Union[DomainInterval, TileInterval, IndexWithExtent]] = {}
        for axis, interval in self.intervals.items():
            if isinstance(interval, DomainInterval) and axis in tile_sizes:
                if axis == Axis.K:
                    res_intervals[axis] = TileInterval(
                        axis=axis,
                        tile_size=tile_sizes[axis],
                        start_offset=0,
                        end_offset=0,
                        domain_limit=interval.end,
                    )
                else:
                    assert (
                        interval.start.level == common.LevelMarker.START
                        and interval.end.level == common.LevelMarker.END
                    )
                    res_intervals[axis] = TileInterval(
                        axis=axis,
                        tile_size=tile_sizes[axis],
                        start_offset=interval.start.offset,
                        end_offset=interval.end.offset,
                        domain_limit=AxisBound(axis=axis, level=common.LevelMarker.END, offset=0),
                    )
            else:
                res_intervals[axis] = interval
        return GridSubset(intervals=res_intervals)

    def union(self, other):
        assert list(self.axes()) == list(other.axes())
        intervals = dict()
        for axis in self.axes():
            interval1 = self.intervals[axis]
            interval2 = other.intervals[axis]
            if isinstance(interval1, DomainInterval) and isinstance(interval2, DomainInterval):
                intervals[axis] = DomainInterval.union(interval1, interval2)
            elif isinstance(interval1, TileInterval) and isinstance(interval2, TileInterval):
                intervals[axis] = TileInterval.union(interval1, interval2)
            elif isinstance(interval1, IndexWithExtent) and isinstance(interval2, IndexWithExtent):
                intervals[axis] = interval1.union(interval2)
            else:
                assert (
                    isinstance(interval2, (TileInterval, DomainInterval))
                    and isinstance(interval1, IndexWithExtent)
                ) or (
                    isinstance(interval1, (TileInterval, DomainInterval))
                    and isinstance(interval2, IndexWithExtent)
                )
                intervals[axis] = (
                    interval1
                    if isinstance(interval1, (TileInterval, DomainInterval))
                    else interval2
                )
        return GridSubset(intervals=intervals)


class FieldAccessInfo(eve.Node):
    grid_subset: GridSubset
    global_grid_subset: GridSubset
    dynamic_access: bool = False
    variable_offset_axes: List[Axis] = eve.field(default_factory=list)

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic_access or len(self.variable_offset_axes) > 0

    def axes(self):
        yield from self.grid_subset.axes()

    @property
    def shape(self):
        return self.grid_subset.shape

    @property
    def overapproximated_shape(self):
        return self.grid_subset.overapproximated_shape

    def apply_iteration(self, grid_subset: GridSubset):
        res_intervals = dict(self.grid_subset.intervals)
        for axis, field_interval in self.grid_subset.intervals.items():
            if axis in grid_subset.intervals:
                grid_interval = grid_subset.intervals[axis]
                assert isinstance(field_interval, IndexWithExtent)
                extent = field_interval.extent
                if isinstance(grid_interval, DomainInterval):
                    if axis in self.global_grid_subset.intervals:
                        res_intervals[axis] = self.global_grid_subset.intervals[axis]
                    else:
                        res_intervals[axis] = DomainInterval(
                            start=AxisBound(
                                axis=axis,
                                level=grid_interval.start.level,
                                offset=grid_interval.start.offset + extent[0],
                            ),
                            end=AxisBound(
                                axis=axis,
                                level=grid_interval.end.level,
                                offset=grid_interval.end.offset + extent[1],
                            ),
                        )
                elif isinstance(grid_interval, TileInterval):
                    res_intervals[axis] = TileInterval(
                        axis=axis,
                        tile_size=grid_interval.tile_size,
                        start_offset=grid_interval.start_offset + extent[0],
                        end_offset=grid_interval.end_offset + extent[1],
                        domain_limit=grid_interval.domain_limit,
                    )
                else:
                    assert field_interval.value == grid_interval.value
                    extent = (
                        min(extent) + grid_interval.extent[0],
                        max(extent) + grid_interval.extent[1],
                    )
                    res_intervals[axis] = IndexWithExtent(
                        axis=axis, value=field_interval.value, extent=extent
                    )
        return FieldAccessInfo(
            grid_subset=GridSubset(intervals=res_intervals),
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
            global_grid_subset=self.global_grid_subset,
        )

    def union(self, other: "FieldAccessInfo"):
        grid_subset = self.grid_subset.union(other.grid_subset)
        global_subset = self.global_grid_subset.union(other.global_grid_subset)
        variable_offset_axes = [
            axis
            for axis in Axis.dims_3d()
            if axis in self.variable_offset_axes or axis in other.variable_offset_axes
        ]
        return FieldAccessInfo(
            grid_subset=grid_subset,
            dynamic_access=self.dynamic_access or other.dynamic_access,
            variable_offset_axes=variable_offset_axes,
            global_grid_subset=global_subset,
        )

    def clamp_full_axis(self, axis):
        grid_subset = GridSubset(intervals=self.grid_subset.intervals)
        interval = self.grid_subset.intervals[axis]
        full_interval = DomainInterval(
            start=AxisBound(level=common.LevelMarker.START, offset=0, axis=axis),
            end=AxisBound(level=common.LevelMarker.END, offset=0, axis=axis),
        )
        res_interval = DomainInterval.union(
            full_interval,
            self.global_grid_subset.intervals.get(axis, full_interval),
        )
        if isinstance(interval, DomainInterval):
            interval_union = DomainInterval.union(interval, res_interval)
            grid_subset.intervals[axis] = interval_union
        else:
            grid_subset.intervals[axis] = res_interval
            grid_subset = grid_subset.set_interval(axis, res_interval)
        return FieldAccessInfo(
            grid_subset=grid_subset,
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
            global_grid_subset=self.global_grid_subset,
        )

    def untile(self, tile_axes: Sequence[Axis]) -> "FieldAccessInfo":
        res_intervals = {}
        for axis, interval in self.grid_subset.intervals.items():
            if isinstance(interval, TileInterval) and axis in tile_axes:
                res_intervals[axis] = self.global_grid_subset.intervals[axis]
            else:
                res_intervals[axis] = interval
        return FieldAccessInfo(
            grid_subset=GridSubset(intervals=res_intervals),
            global_grid_subset=self.global_grid_subset,
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
        )

    def restricted_to_index(self, axis: Axis, extent: Tuple[int, int] = (0, 0)):
        return FieldAccessInfo(
            grid_subset=self.grid_subset.restricted_to_index(axis=axis, extent=extent),
            global_grid_subset=self.global_grid_subset,
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
        )


class Memlet(eve.Node):
    field: eve.Coerced[eve.SymbolRef]
    access_info: FieldAccessInfo
    connector: eve.Coerced[eve.SymbolRef]
    is_read: bool
    is_write: bool

    def union(self, other):
        assert self.field == other.field
        return Memlet(
            field=self.field,
            access_info=self.access_info.union(other.access_info),
            connector=self.field,
            is_read=self.is_read or other.is_read,
            is_write=self.is_write or other.is_write,
        )

    def remove_read(self):
        return Memlet(
            field=self.field,
            access_info=self.access_info,
            connector=self.connector,
            is_read=False,
            is_write=self.is_write,
        )

    def remove_write(self):
        return Memlet(
            field=self.field,
            access_info=self.access_info,
            connector=self.connector,
            is_read=self.is_read,
            is_write=False,
        )


class Decl(LocNode):
    name: eve.Coerced[eve.SymbolName]
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    strides: Tuple[Union[int, str], ...]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)
    access_info: FieldAccessInfo
    storage: StorageType

    @property
    def shape(self):
        access_info = self.access_info
        for axis in self.access_info.variable_offset_axes:
            access_info = access_info.clamp_full_axis(axis)
        ijk_shape = access_info.grid_subset.shape
        return ijk_shape + tuple(self.data_dims)

    def axes(self):
        yield from self.access_info.grid_subset.axes()

    @property
    def is_dynamic(self) -> bool:
        return self.access_info.is_dynamic

    def with_set_access_info(self, access_info: FieldAccessInfo) -> "FieldDecl":
        return FieldDecl(
            name=self.name,
            dtype=self.dtype,
            strides=self.strides,
            data_dims=self.data_dims,
            access_info=access_info,
        )


class Literal(common.Literal, Expr):
    pass


class ScalarAccess(common.ScalarAccess, Expr):
    name: eve.Coerced[eve.SymbolRef]


class VariableKOffset(common.VariableKOffset[Expr]):
    pass


class IndexAccess(common.FieldAccess, Expr):
    offset: Optional[Union[common.CartesianOffset, VariableKOffset]]


class AssignStmt(common.AssignStmt[Union[ScalarAccess, IndexAccess], Expr], Stmt):

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class MaskStmt(Stmt):
    mask: Expr
    body: List[Stmt]

    @datamodels.validator("mask")
    def mask_is_boolean_field_expr(self, attribute: datamodels.Attribute, v: Expr) -> None:
        if v.dtype != common.DataType.BOOL:
            raise ValueError("Mask must be a boolean expression.")


class HorizontalRestriction(common.HorizontalRestriction[Stmt], Stmt):
    pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class While(common.While[Stmt, Expr], Stmt):
    pass


class ScalarDecl(Decl):
    pass


class LocalScalarDecl(ScalarDecl):
    pass


class SymbolDecl(ScalarDecl):
    def to_dace_symbol(self):
        return get_dace_symbol(self.name, self.dtype)


class Temporary(FieldDecl):
    pass


def _unique_connectors(*, field: str) -> datamodels.FieldValidator:
    def _validator(
        self: "ComputationNode", attribute: datamodels.Attribute, node: List[Memlet]
    ) -> None:
        conns: Dict[eve.SymbolRef, Set[eve.SymbolRef]] = {}
        for memlet in node:
            conns.setdefault(memlet.field, set())
            if memlet.connector in conns[memlet.field]:
                raise ValueError(f"Found multiple Memlets for connector '{memlet.connector}'")
            conns[memlet.field].add(memlet.connector)

    return datamodels.validator(field)(_validator)


class ComputationNode(LocNode):
    # mapping connector names to tuple of field name and access info
    read_memlets: List[Memlet]
    write_memlets: List[Memlet]

    unique_write_connectors = _unique_connectors(field="write_memlets")
    unique_read_connectors = _unique_connectors(field="write_memlets")

    @property
    def read_fields(self):
        return set(ml.field for ml in self.read_memlets)

    @property
    def write_fields(self):
        return set(ml.field for ml in self.write_memlets)

    @property
    def input_connectors(self):
        return set(ml.connector for ml in self.read_memlets)

    @property
    def output_connectors(self):
        return set(ml.connector for ml in self.write_memlets)


class IterationNode(eve.Node):
    grid_subset: GridSubset


class Tasklet(ComputationNode, IterationNode, eve.SymbolTableTrait):
    decls: List[LocalScalarDecl]
    stmts: List[Stmt]
    grid_subset: GridSubset = GridSubset.single_gridpoint()


class DomainMap(ComputationNode, IterationNode):
    index_ranges: List[Range]
    schedule: MapSchedule
    computations: List[Union[Tasklet, "DomainMap", "NestedSDFG"]]


class ComputationState(IterationNode):
    computations: List[Union[Tasklet, DomainMap]]


class DomainLoop(IterationNode, ComputationNode):
    axis: Axis
    index_range: Range
    loop_states: List[Union[ComputationState, "DomainLoop"]]


class NestedSDFG(ComputationNode, eve.SymbolTableTrait):
    label: eve.Coerced[eve.SymbolRef]
    field_decls: List[FieldDecl]
    symbol_decls: List[SymbolDecl]
    states: List[Union[DomainLoop, ComputationState]]


# There are circular type references with string placeholders. These statements let pydantic resolve those.
DomainMap.update_forward_refs()  # type: ignore[attr-defined]
DomainLoop.update_forward_refs()  # type: ignore[attr-defined]
