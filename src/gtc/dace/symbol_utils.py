from functools import lru_cache
from typing import TYPE_CHECKING

import dace
import numpy as np

import eve
from gtc import common


if TYPE_CHECKING:
    import gtc.daceir as dcir


def data_type_to_dace_typeclass(data_type):
    dtype = np.dtype(common.data_type_to_typestr(data_type))
    return dace.dtypes.typeclass(dtype.type)


def get_axis_bound_str(axis_bound, var_name):
    from gtc.common import LevelMarker

    if axis_bound is None:
        return ""
    elif axis_bound.level == LevelMarker.END:
        return f"{var_name}{axis_bound.offset:+d}"
    else:
        return f"{axis_bound.offset}"


def get_axis_bound_dace_symbol(axis_bound: "dcir.AxisBound"):
    from gtc.common import LevelMarker

    if axis_bound is None:
        return

    elif axis_bound.level == LevelMarker.END:
        return axis_bound.axis.domain_dace_symbol() + axis_bound.offset
    else:
        return axis_bound.offset


def get_axis_bound_diff_str(axis_bound1, axis_bound2, var_name: str):

    if axis_bound1 <= axis_bound2:
        axis_bound1, axis_bound2 = axis_bound2, axis_bound1
        sign = "-"
    else:
        sign = ""

    if axis_bound1.level != axis_bound2.level:
        var = var_name
    else:
        var = ""
    return f"{sign}({var}{axis_bound1.offset-axis_bound2.offset:+d})"


@lru_cache(maxsize=None)
def get_dace_symbol(name: eve.SymbolRef, dtype: common.DataType = common.DataType.INT32):
    return dace.symbol(name, dtype=data_type_to_dace_typeclass(dtype))
