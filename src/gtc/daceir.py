from typing import Dict, List, Tuple, Union

import dace

from eve import Int, IntEnum, Node, Str, StrEnum
from gt4py import definitions as gt_def
from gtc import common, oir
from gtc.common import LocNode
from gtc.oir import LocalScalar, Stmt

from .dace.utils import get_axis_bound_diff_str, get_axis_bound_str


class Axis(StrEnum):
    I = "I"  # noqa: E741 ambiguous variable name 'I'
    J = "J"
    K = "K"

    def iteration_symbol(self):
        return "__" + self.lower()

    def domain_symbol(self):
        return "__" + self.upper()

    def tile_symbol(self):
        return "__tile_" + self.lower()

    @staticmethod
    def dims_3d():
        yield from [Axis.I, Axis.J, Axis.K]

    @staticmethod
    def dims_horizontal():
        yield from [Axis.I, Axis.J]

    @staticmethod
    def horizontal_axes():
        yield from [Axis.I, Axis.J]

    def to_idx(self):
        return [Axis.I, Axis.J, Axis.K].index(self)


class MapSchedule(IntEnum):
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


class StorageType(IntEnum):
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


class IndexWithExtent(Node):
    axis: Axis
    value: Union[AxisBound, Int, Str]
    extent: Tuple[int, int]

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

    def shifted(self, offset):
        extent = self.extent[0] + offset, self.extent[1] + offset
        return IndexWithExtent(axis=self.axis, value=self.value, extent=extent)


class DomainInterval(Node):
    start: AxisBound
    end: AxisBound

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


class TileInterval(Node):
    axis: Axis
    start_offset: int
    end_offset: int
    tile_size: int
    domain_limit: str

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
            domain_limit=self.domain_limit,
            tile_symbol=self.axis.tile_symbol(),
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


class Range(Node):
    var: Str
    start: Union[Str, Int, AxisBound]
    end: Union[Str, Int, AxisBound]
    stride: Union[Str, Int] = 1

    def to_ndrange(self):
        if isinstance(self.start, AxisBound):
            start = get_axis_bound_str(self.start, self.start.axis.domain_symbol())
        else:
            start = str(self.start)
        if isinstance(self.end, AxisBound):
            end = get_axis_bound_str(self.end, self.end.axis.domain_symbol())
        else:
            end = str(self.end)
        return {self.var: f"{start}:{end}:{self.stride}"}

    @classmethod
    def from_axis_and_interval(
        cls, axis: Axis, interval: Union[DomainInterval, TileInterval], stride=1
    ):
        if isinstance(interval, (oir.Interval, DomainInterval)):
            start = get_axis_bound_str(interval.start, axis.domain_symbol())
            end = get_axis_bound_str(interval.end, axis.domain_symbol())
        else:
            start, end = interval.idx_range
        return cls(
            var=axis.iteration_symbol(),
            start=start,
            end=end,
            stride=stride,
        )


class GridSubset(Node):
    intervals: Dict[Axis, Union[DomainInterval, TileInterval, IndexWithExtent]]

    def __iter__(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield self.intervals[axis]

    def items(self):
        for axis in Axis.dims_3d():
            if axis in self.intervals:
                yield axis, self.intervals[axis]

    @classmethod
    def single_gridpoint(cls):
        return cls(intervals={axis: IndexWithExtent.from_axis(axis) for axis in Axis.dims_3d()})

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
        self, axis: Axis, interval: Union[DomainInterval, IndexWithExtent, oir.Interval]
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
    def from_gt4py_extent(cls, extent: gt_def.Extent):
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
        cls, interval: Union[oir.Interval, DomainInterval, IndexWithExtent], axis: Axis
    ):
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
        res_intervals = dict()
        for axis, interval in self.intervals.items():
            if isinstance(interval, DomainInterval) and axis in tile_sizes:
                if axis == Axis.K:
                    res_intervals[axis] = TileInterval(
                        axis=axis,
                        tile_size=tile_sizes[axis],
                        start_offset=0,
                        end_offset=0,
                        domain_limit=str(interval.end),
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
                        domain_limit=axis.domain_symbol(),
                    )
            else:
                res_intervals[axis] = interval
        return GridSubset(intervals=res_intervals)

    # def untile(self, tile_axes: List[Axis]):
    #     res_intervals = dict()
    #     for axis, interval in self.intervals.items():
    #         if isinstance(interval, TileInterval) and axis in tile_axes:
    #             res_intervals[axis] = DomainInterval(
    #                 start=AxisBound(
    #                     axis=axis,
    #                     level=common.LevelMarker.START,
    #                     offset=interval.start_offset,
    #                 ),
    #                 end=AxisBound(
    #                     axis=axis,
    #                     level=common.LevelMarker.END,
    #                     offset=interval.end_offset,
    #                 ),
    #             )
    #         else:
    #             res_intervals[axis] = interval
    #     return GridSubset(intervals=res_intervals)

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


class FieldAccessInfo(Node):
    grid_subset: GridSubset
    global_grid_subset: GridSubset
    dynamic_access: bool = False
    variable_offset_axes: List[Axis] = []

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
        if isinstance(interval, DomainInterval):
            interval_union = DomainInterval.union(interval, full_interval)
            grid_subset.intervals[axis] = interval_union
        else:
            grid_subset.intervals[axis] = full_interval
            grid_subset = grid_subset.set_interval(axis, full_interval)
        return FieldAccessInfo(
            grid_subset=grid_subset,
            dynamic_access=self.dynamic_access,
            variable_offset_axes=self.variable_offset_axes,
            global_grid_subset=self.global_grid_subset,
        )

    def untile(self, tile_axes: List[Axis]):
        res_intervals = dict()
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


class FieldDecl(Node):
    name: Str
    dtype: common.DataType
    strides: List[Union[Int, Str]]
    data_dims: List[Int]
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


class ComputationNode(Node):
    # mapping connector names to tuple of field name and access info
    read_accesses: Dict[str, FieldAccessInfo]
    write_accesses: Dict[str, FieldAccessInfo]


class IterationNode(Node):
    grid_subset: GridSubset


class NestedSDFGNode(ComputationNode):
    field_decls: Dict[str, FieldDecl]
    symbols: Dict[str, common.DataType]
    name_map: Dict[str, str]


class Tasklet(IterationNode, ComputationNode, LocNode):
    stmts: List[Union[LocalScalar, Stmt]]
    grid_subset: GridSubset = GridSubset.single_gridpoint()
    name_map: Dict[str, str]


class DomainMap(ComputationNode, IterationNode):
    index_ranges: List[Range]
    schedule: MapSchedule
    computations: List[Union[Tasklet, "DomainMap", "StateMachine"]]


class ComputationState(IterationNode):
    computations: List[Union[Tasklet, DomainMap]]


class CopyState(ComputationNode, IterationNode):
    name_map: Dict[str, str]


class DomainLoop(IterationNode, ComputationNode):
    axis: Axis
    index_range: Range
    loop_states: List[Union[ComputationState, CopyState, "DomainLoop"]]


class StateMachine(NestedSDFGNode):
    label: str
    states: List[Union[DomainLoop, CopyState, ComputationState]]


DomainMap.update_forward_refs()
DomainLoop.update_forward_refs()
