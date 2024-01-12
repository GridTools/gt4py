from . import types as ts_f
from gt4py.next.type_system_2 import types as ts, utils as ts_utils

from gt4py.next import common as gtx_common


def is_field_type_local(ty: ts_f.FieldType):
    return any(dim.kind == gtx_common.DimensionKind.LOCAL for dim in ty.dimensions)


def contains_local_field(type_: ts.Type) -> bool:
    """Determine if there is a local field among the elements of `type_`."""
    return any(
        isinstance(t, ts_f.FieldType) and is_field_type_local(t) for t in ts_utils.flatten_tuples(type_)
    )
