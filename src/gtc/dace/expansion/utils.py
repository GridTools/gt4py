from typing import Union

import dace
import dace.data
import dace.library
import dace.subsets

from gtc import common
from gtc import daceir as dcir
from gtc.dace.utils import get_axis_bound_str


def add_origin(
    access_info: dcir.FieldAccessInfo,
    subset: Union[dace.subsets.Range, str],
    add_for_variable=False,
):
    if isinstance(subset, str):
        subset = dace.subsets.Range.from_string(subset)
    origin_strs = []
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes and not add_for_variable:
            origin_strs.append(str(0))
        elif add_for_variable:
            clamped_interval = access_info.clamp_full_axis(axis).grid_subset.intervals[axis]
            origin_strs.append(
                f"-({get_axis_bound_str(clamped_interval.start, axis.domain_symbol())})"
            )
        else:
            interval = access_info.grid_subset.intervals[axis]
            if isinstance(interval, dcir.DomainInterval):
                origin_strs.append(f"-({get_axis_bound_str(interval.start, axis.domain_symbol())})")
            elif isinstance(interval, dcir.TileInterval):
                origin_strs.append(f"-({interval.axis.tile_symbol()}{interval.start_offset:+d})")
            else:
                assert isinstance(interval, dcir.IndexWithExtent)
                origin_strs.append(f"-({interval.value}{interval.extent[0]:+d})")

    sym = dace.symbolic.pystr_to_symbolic
    res_ranges = []
    for i, axis in enumerate(access_info.axes()):
        rng = subset.ranges[i]
        orig = origin_strs[axis.to_idx()]
        res_ranges.append((sym(f"({rng[0]})+({orig})"), sym(f"({rng[1]})+({orig})"), rng[2]))
    return dace.subsets.Range(res_ranges)


def get_dace_debuginfo(node: common.LocNode):

    if node.loc is not None:
        return dace.dtypes.DebugInfo(
            node.loc.line,
            node.loc.column,
            node.loc.line,
            node.loc.column,
            node.loc.source,
        )
    else:
        return dace.dtypes.DebugInfo(0)
