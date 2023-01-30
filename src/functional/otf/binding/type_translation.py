from typing import Any, Optional

from functional.iterator.embedded import IndexField
from functional.otf.binding import type_specifications as ts_binding
from functional.type_system import type_specifications as ts, type_translation


def from_type_hint(
    type_hint: Any,
    *,
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> ts.TypeSpec:
    return type_translation.from_type_hint(type_hint, globalns=globalns, localns=localns)


def from_value(value: Any) -> ts.TypeSpec:
    if isinstance(value, IndexField):
        dtype = type_translation.from_type_hint(value.dtype.type)
        assert isinstance(dtype, ts.ScalarType)
        return ts_binding.IndexFieldType(axis=value.axis, dtype=dtype)
    else:
        return type_translation.from_value(value)
