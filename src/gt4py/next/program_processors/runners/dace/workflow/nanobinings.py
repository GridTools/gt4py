# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Mapping

import dace
import numpy as np

from gt4py.next.embedded import nd_array_field
from gt4py.next.iterator import ir
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon
from gt4py.next.type_system import type_specifications as ts


def make_user_args(
    sdfg: dace.SDFG,
    prog: ir.Program,
    use_metrics: bool,
    offset_provider: Mapping[str, Any],
) -> list[Any]:
    sdfg_arglist = sdfg.arglist().copy()

    user_args = [
        _make_user_args(
            sdfg=sdfg,
            sdfg_arglist=sdfg_arglist,
            param_name=param.id,
            param_type=param.type,  # type: ignore[arg-type]  # Never `None`
        )
        for param in prog.params
    ]

    user_args.extend(
        _handle_connectivities(sdfg_arglist=sdfg_arglist, offset_provider=offset_provider)
    )

    if use_metrics:
        user_args.extend(
            [gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL, gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME]
        )

    return user_args


def _make_user_args(
    sdfg: dace.SDFG,
    sdfg_arglist: Mapping[str, dace.data.Data],
    param_name: str,
    param_type: ts.TypeSpec,
) -> Any | tuple[Any, ...]:

    if param_type is None:
        ValueError(f"Expected that parameter `{param_name}` carries a type.")

    if isinstance(param_type, ts.TupleType):
        return tuple(
            _make_user_args(
                sdfg=sdfg,
                sdfg_arglist=sdfg_arglist,
                param_name=f"{param_name}_{i}",
                param_type=tuple_arg_type,
            )
            for i, tuple_arg_type in enumerate(param_type.types)
        )

    elif param_name not in sdfg_arglist:
        # There are two reasons for this case:
        #   1) The argument is a symbol/scalar that is not used in the generated code.
        #   2) The argument was demoted, see `demote_fields` argument of `gt_auto_optimize()`
        #       and was not put back.

        if (isinstance(param_type, ts.FieldType) and len(param_type.dims) == 0) or isinstance(
            param_type, ts.ScalarType
        ):
            alt_param_name, _ = sdfg.add_scalar(
                f"__gt4py_unused_{param_name}",
                dtype=dace.dtypes.pyobject(),
                transient=False,
                find_new_name=True,
            )
            return alt_param_name

        else:
            alt_param_name, _ = sdfg.add_scalar(
                f"__gt4py_unused_{param_name}_value",
                dtype=dace.dtypes.pyobject(),
                transient=False,
                find_new_name=True,
            )
            alt_param_origin_name, _ = sdfg.add_scalar(
                f"__gt4py_unused_{param_name}_origin",
                dtype=dace.dtypes.pyobject(),
                transient=False,
                find_new_name=True,
            )
            return alt_param_name, alt_param_origin_name

    elif isinstance(param_type, ts.FieldType):
        if param_name not in sdfg.arrays:
            ValueError(f"Not not find array parameter `{param_name}` in the SDFG array registry.")
        if param_name not in sdfg_arglist:
            ValueError(f"Did not find array parameter `{param_name}` in the SDFG argument list.")

        sdfg_arg_desc: dace.data.Data = sdfg.arrays[param_name]
        if sdfg_arg_desc.transient:
            ValueError(f"GT4Py parameter `{param_name}` is a transient.")

        if len(param_type.dims) == 0:
            # Pass zero-dimensional fields as scalars. In call path it will be `as_scalar()`ed.
            assert isinstance(sdfg_arg_desc, dace.data.Scalar)
            return param_name

        else:
            # Full array.
            #  It will be passed as `(array_name_in_sdfg, ORIGINES)`. Note that the origins
            #  are included even if they are known at compile time, this is for compatibility
            #  with GTFN. Also important, in that case we will add the symbol back, because we
            #  have to refer to it.
            origins: list[str] = []
            for dim in param_type.dims:
                rstart = str(gtx_dace_args.range_start_symbol(param_name, dim))
                if rstart in sdfg_arglist:
                    assert rstart in sdfg.symbols
                    assert rstart not in sdfg.arrays
                    origins.append(rstart)
                else:
                    # For certain reason the dimension parameter is not needed and thus not
                    #  included. In this case it can not be an SDFG array nor a symbol. For
                    #  compatibility with GTFN we have to pass it as an argument and thus
                    #  put it back.
                    # TODO(phimuell): Give it a name that is different.
                    assert rstart not in sdfg.symbols
                    assert rstart not in sdfg.arrays

                    # TODO(phimuell): Find out what the true type should be.
                    sdfg.add_symbol(rstart, dace.dtypes.int32, find_new_name=False)
                    origins.append(rstart)

            return (param_name, tuple(origins))

    elif isinstance(param_type, ts.ScalarType):
        # A scalar name, so simply return the name of the parameter.
        if not (param_name in sdfg.arrays or param_name in sdfg.symbols):
            ValueError(
                f"Not not find scalar parameter `{param_name}` in the SDFG array/symbol registry."
            )
        if param_name not in sdfg_arglist:
            ValueError(f"Did not find scalar parameter `{param_name}` in the SDFG argument list.")
        return param_name

    else:
        raise ValueError(f"Parameter `{param_name}` had unexpected type `{param_type}`")


def _handle_connectivities(
    sdfg_arglist: Mapping[str, dace.data.Data],
    offset_provider: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            arg_name
            for arg_name in sdfg_arglist.keys()
            if gtx_dace_args.is_connectivity_identifier(arg_name)
        )
    )


#######################################
#   CALLING


def convert_arg(arg: Any) -> Any:
    # NOTE: this function is on the hot path and needs to have minimal overhead.
    # NOTE: Copied from GTFN path, but we need `to_scalar()` for zero dimensional array.
    if (origin := getattr(arg, "__gt_origin__", None)) is not None:
        # `Field` is the most likely case, we use `__gt_origin__` as the property is needed anyway
        # and (currently) uniquely identifies a `NDArrayField` (which is the only supported `Field`)
        assert isinstance(arg, nd_array_field.NdArrayField)
        if len(origin) == 0:
            return arg.as_scalar()
        else:
            return arg.ndarray, origin
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, np.bool_):
        # nanobind does not support implicit conversion of `np.bool` to `bool`
        return bool(arg)

    # TODO(havogt): if this function still appears in profiles,
    # we should avoid going through the previous isinstance checks for detecting a scalar.
    # E.g. functools.cache on the arg type, returning a function that does the conversion
    return arg
