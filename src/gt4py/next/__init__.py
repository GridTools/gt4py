# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
GT4Py-NEXT - Performance portable and composable weather & climate stencils.

This module deviates from the project coding style (Google Python style) in the following way:

[style guide link](https://google.github.io/styleguide/pyguide.html#22-imports):

Fnctions and classes are imported from other modules in order to explicitly re-export them
to create a streamlined user experience. `from module import *` can be used but only if the
module in question is a submodule, defines `__all__` and exports many public API objects.
"""

# ruff: noqa: F401
import importlib
from typing import TYPE_CHECKING, Any

from .._core.definitions import CUPY_DEVICE_TYPE, Device, DeviceType, is_scalar_type
from . import common, ffront, iterator, program_processors, typing
from .common import (
    CartesianConnectivity,
    Connectivity,
    Dimension,
    DimensionKind,
    Dims,
    Domain,
    Field,
    GridType,
    UnitRange,
    as_non_staggered,
    domain,
    flip_staggered,
    is_staggered,
    unit_range,
)
from .constructors import FieldConstructor, as_connectivity, as_field, empty, full, ones, zeros
from .embedded import (  # Just for registering field implementations
    nd_array_field as _nd_array_field,
)
from .ffront import fbuiltins
from .ffront.decorator import field_operator, program, scan_operator
from .ffront.fbuiltins import (
    FieldOffset,
    IndexType,
    abs,  # noqa: A004 # shadowing
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    astype,
    bool,  # noqa: A004 # shadowing
    broadcast,
    cbrt,
    ceil,
    cos,
    cosh,
    exp,
    float,  # noqa: A004 # shadowing
    float32,
    float64,
    floor,
    fmod,
    gamma,
    int,  # noqa: A004 # shadowing
    int8,
    int16,
    int32,
    int64,
    isfinite,
    isinf,
    isnan,
    log,
    max_over,
    maximum,
    min_over,
    minimum,
    neg,
    neighbor_sum,
    power,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    tuple,  # noqa: A004 # shadowing
    uint8,
    uint16,
    uint32,
    uint64,
    where,
)
from .otf.compiled_program import wait_for_compilation
from .program_processors.runners.gtfn import run_gtfn as gtfn_cpu, run_gtfn_gpu as gtfn_gpu
from .program_processors.runners.roundtrip import default as itir_python


if TYPE_CHECKING:
    # The DaCe backends are exported lazily (see `__getattr__` below), but type
    # checkers and IDEs need to see them as regular module attributes.
    from .program_processors.runners.dace import run_dace_cpu as dace_cpu, run_dace_gpu as dace_gpu


__all__ = [  # noqa: RUF022 [unsorted-dunder-all]
    # submodules
    "common",
    "ffront",
    "iterator",
    "program_processors",
    "typing",
    # from _core.definitions
    "CUPY_DEVICE_TYPE",
    "Device",
    "DeviceType",
    "is_scalar_type",
    # from common
    "Dimension",
    "DimensionKind",
    "Dims",
    "Field",
    "CartesianConnectivity",
    "Connectivity",
    "GridType",
    "domain",
    "Domain",
    "unit_range",
    "UnitRange",
    "is_staggered",
    "flip_staggered",
    "as_non_staggered",
    # from constructors
    "FieldConstructor",
    "empty",
    "zeros",
    "ones",
    "full",
    "as_field",
    "as_connectivity",
    # from ffront
    "FieldOffset",
    "field_operator",
    "program",
    "scan_operator",
    # from otf
    "wait_for_compilation",
    # from program_processor
    "gtfn_cpu",
    "gtfn_gpu",
    "dace_cpu",
    "dace_gpu",
    "itir_python",
    # from fbuiltins
    "IndexType",
    "abs",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "astype",
    "bool",
    "broadcast",
    "cbrt",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "float",
    "float32",
    "float64",
    "floor",
    "fmod",
    "gamma",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "isfinite",
    "isinf",
    "isnan",
    "log",
    "max_over",
    "min_over",
    "maximum",
    "minimum",
    "neg",
    "neighbor_sum",
    "power",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "tuple",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "where",
]

assert not (diff := set(fbuiltins.__all__) - set(__all__)), (
    f"public fbuiltins member(s) not exported: {diff}"
)


# Public names resolved lazily: public name -> (module, attribute).
#
# Loading the DaCe backends imports the DaCe toolchain modules and builds the backend
# objects, which adds noticeable time to `import gt4py.next` and is not needed unless
# a DaCe backend is actually used. The resolved value is cached in the module globals,
# so `__getattr__` is only called once per name.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "dace_cpu": (".program_processors.runners.dace", "run_dace_cpu"),
    "dace_gpu": (".program_processors.runners.dace", "run_dace_gpu"),
}


def __getattr__(name: str) -> Any:
    if (target := _LAZY_EXPORTS.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
