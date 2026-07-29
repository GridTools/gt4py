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
        _add_connectivities_to_user_args(sdfg_arglist=sdfg_arglist, offset_provider=offset_provider)
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
            # Passed as 1 parameter (must be consistent with `convert_arg()`.
            return ""
        else:
            # Arrays are passed as a tuple consisting of the array and a tuple for the origin.
            # TODO(phimuell): Find out if also a single ignore slot is okay.
            assert isinstance(param_type, ts.FieldType)
            return ("", ("") * len(param_type.dims))

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
            found_needed_symbol = False
            for dim in param_type.dims:
                rstart = str(gtx_dace_args.range_start_symbol(param_name, dim))
                if rstart in sdfg_arglist:
                    assert rstart in sdfg.symbols
                    assert rstart not in sdfg.arrays
                    origins.append(rstart)
                    found_needed_symbol = True
                else:
                    # For certain reason the dimension parameter is not needed and thus not
                    #  included. For compatibility with GTFN we have to provide it, but ignore it.
                    #  Note that it could still be inside the symbols table.
                    origins.append("")

            return (param_name, tuple(origins) if found_needed_symbol else "")

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


def _add_connectivities_to_user_args(
    sdfg_arglist: Mapping[str, dace.data.Data],
    offset_provider: Mapping[str, Any],
) -> tuple:
    # This is fully call compatible with GTFN and also depends on that the order of the dict is stable.
    # TODO(phimuell): Find out if a single ignore `""` in the full ignore case is enough.
    return tuple(
        (offset_arg_name, ("", "")) if offset_arg_name in sdfg_arglist else ("", ("", ""))
        for offset_arg_name in map(gtx_dace_args.connectivity_identifier, offset_provider.keys())
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
            # Difference between the two things.
            return arg.ndarray, tuple(-s for s in origin)
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, np.bool_):
        # nanobind does not support implicit conversion of `np.bool` to `bool`
        return bool(arg)

    # TODO(havogt): if this function still appears in profiles,
    # we should avoid going through the previous isinstance checks for detecting a scalar.
    # E.g. functools.cache on the arg type, returning a function that does the conversion
    return arg


##########################################################
#   CODE GEN


def write_convert_code(
    update_function_name: str,
    prog: ir.Program,
    use_metrics: bool,
    offset_provider: Mapping[str, Any],
    back_end: str,
    offset_provider_name: str = "offset_provider",
) -> str:
    back_end = back_end.lower()
    assert back_end in ["gtfn", "dace"]

    function_args: list[str] = []
    unpack_variables: list[str] = []

    def _unpack_tuple(arg_name: str, tuple_len: int) -> list[str]:
        tuple_args = [f"{arg_name}_{i}" for i in range(tuple_len)]
        unpack_variables.append(f"{', '.join(tuple_args)}, = {arg_name}")
        return tuple_args

    def _extract_arg(var_name: str, param_type: ts.TypeSpec) -> str:
        if isinstance(param_type, ts.TupleType):
            member_names = _unpack_tuple(var_name, len(param_type.types))
            tuple_expand_code = "("
            for member_name, tuple_arg_type in zip(member_names, param_type.types):
                tuple_expand_code += _extract_arg(member_name, tuple_arg_type)
            tuple_expand_code += ")"
            return tuple_expand_code

        elif isinstance(param_type, ts.FieldType):
            if len(param_type.dims) == 0:
                # GTFN wants a field, dace a scalar.
                return f"{var_name}," if back_end == "gtfn" else f"{var_name}.as_scalar(), "
            else:
                # Full array, the differences is where the origin comes from.
                origin_prop = "__gt_origin__" if back_end == "gtfn" else "__dace_origin__"
                return f"({var_name}.ndarray, {var_name}.{origin_prop}), "
        elif isinstance(param_type, ts.ScalarType):
            if param_type.kind == ts.ScalarKind.BOOL:
                return f"bool({var_name}), "
            else:
                return f"{var_name}, "
        else:
            raise ValueError(f"Parameter `{var_name}` had unexpected type `{param_type}`")

    # Now make the arguments.
    tuple_def_code = "("
    for param in prog.params:
        arg_name = f"_auto_expand_name_{param.id}"
        function_args.append(arg_name)
        assert param.type is not None  # To make mypy happy.
        tuple_def_code += _extract_arg(var_name=arg_name, param_type=param.type)

    # Followed by the offset tables, note that this assumes that the order is stable, because we do not check it.
    function_args.append(offset_provider_name)
    for table_name in offset_provider:
        tuple_def_code += f"({offset_provider_name}[{table_name}].ndarray, (0, 0)), "

    # This is currently different and should probably be unified somehow.
    if back_end == "gtfn":
        raise NotImplementedError("Implement me")
    else:
        if use_metrics:
            function_args.extend(
                [gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL, gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME]
            )
            tuple_def_code += f"{gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL}, {gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME}, "

    tuple_def_code += ")"

    # Now writing the code
    code_lines: list[str] = []
    code_lines.append(
        "from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon"
    )
    code_lines.append(f"def {update_function_name}({', '.join(function_args)}):")
    code_lines.extend("\t" + unpack_line for unpack_line in unpack_variables)
    code_lines.append(f"\treturn {tuple_def_code}")

    return "\n".join(code_lines)
