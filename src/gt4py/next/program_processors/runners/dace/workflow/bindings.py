# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import Final, TypeVar

import dace

from gt4py.eve import codegen
from gt4py.next.iterator import builtins as itir_builtins
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils
from gt4py.next.type_system import type_specifications as ts


SrcL = TypeVar("SrcL", bound=languages.SDFG, covariant=True)

FIELD_RANGE_PARAM_RE: Final[re.Pattern] = re.compile(r"^__(.+)_(\d+)_range$")
FIELD_SYMBOL_GT_TYPE: Final[ts.ScalarType] = ts.ScalarType(
    kind=getattr(ts.ScalarKind, itir_builtins.INTEGER_INDEX_BUILTIN.upper())
)

_cb_args: Final[str] = "args"
_cb_device: Final[str] = "device"
_cb_get_stride: Final[str] = "_get_stride"
_cb_sdfg_argtypes: Final[str] = "sdfg_argtypes"
_cb_last_call_args: Final[str] = "last_call_args"


def _update_sdfg_scalar_arg(
    code: codegen.TextBlock, rhs: str, sdfg_arg_desc: dace.data.Data, sdfg_arg_index: int
) -> None:
    """Emit Python code to update a scalar argument in the SDFG arglist."""
    assert isinstance(sdfg_arg_desc, dace.data.Scalar)
    actype = sdfg_arg_desc.dtype.as_ctypes()
    actype_call = f"{actype.__module__}.{actype.__name__}"
    code.append(f"assert isinstance({_cb_last_call_args}[{sdfg_arg_index}], ctypes._SimpleCData)")
    code.append(f"{_cb_last_call_args}[{sdfg_arg_index}] = {actype_call}({rhs})")


def _validate_sdfg_scalar_arg(
    code: codegen.TextBlock, rhs: str, sdfg_arg_desc: dace.data.Data, sdfg_arg_index: int
) -> None:
    """Emit Python asserts to validate a scalar argument in the SDFG arglist."""
    assert isinstance(sdfg_arg_desc, dace.data.Scalar)
    code.append(f"assert isinstance({_cb_last_call_args}[{sdfg_arg_index}], ctypes._SimpleCData)")
    code.append(f"assert {_cb_last_call_args}[{sdfg_arg_index}] == {rhs}")


def _parse_gt_param(
    param_name: str,
    param_type: ts.DataType,
    arg: str,
    code: codegen.TextBlock,
    sdfg_arglist: dict[str, dace.data.Data],
    sdfg_argnames: list[str],
    make_persistent: bool,
) -> None:
    """Emit Python code to update an argument in the SDFG arglist."""
    if isinstance(param_type, ts.TupleType):
        # Special handling of tuples
        if (m := FIELD_RANGE_PARAM_RE.match(param_name)) is not None:
            # Domain range is expressed as a tuple in each dimension
            gt_field_name, dim_index = m[1], int(m[2])
            rstart = gtx_dace_utils.range_start_symbol(gt_field_name, dim_index)
            rstop = gtx_dace_utils.range_stop_symbol(gt_field_name, dim_index)
            for i, tuple_param_name in enumerate([rstart, rstop]):
                tuple_arg = f"{arg}[{i}]"
                tuple_param_type = param_type.types[i]
                assert isinstance(tuple_param_type, ts.ScalarType)
                _parse_gt_param(
                    tuple_param_name,
                    tuple_param_type,
                    tuple_arg,
                    code,
                    sdfg_arglist,
                    sdfg_argnames,
                    make_persistent,
                )
        else:
            # For regular data tuples, each element of the tuple gets a name
            # with an index-based suffix and it is recursively visited.
            for i, tuple_param_type in enumerate(param_type.types):
                tuple_arg = f"{arg}[{i}]"
                tuple_param_name = f"{param_name}_{i}"
                assert isinstance(tuple_param_type, ts.DataType)
                _parse_gt_param(
                    tuple_param_name,
                    tuple_param_type,
                    tuple_arg,
                    code,
                    sdfg_arglist,
                    sdfg_argnames,
                    make_persistent,
                )

    elif param_name not in sdfg_arglist:
        # symbols that are not used are removed from the SDFG arglist
        assert isinstance(param_type, ts.ScalarType)
        assert gtx_dace_utils.is_field_symbol(param_name)

    else:
        sdfg_arg_desc = sdfg_arglist[param_name]
        sdfg_arg_index = sdfg_argnames.index(param_name)

        if isinstance(param_type, ts.FieldType):
            if len(param_type.dims) == 0:
                # Pass zero-dimensional fields as scalars.
                assert isinstance(sdfg_arg_desc, dace.data.Scalar)
                _update_sdfg_scalar_arg(code, f"{arg}.as_scalar()", sdfg_arg_desc, sdfg_arg_index)
            else:
                assert isinstance(sdfg_arg_desc, dace.data.Array)
                code.append(f"assert field_utils.verify_device_field_type({arg}, {_cb_device})")
                code.append(
                    f"assert isinstance({_cb_last_call_args}[{sdfg_arg_index}], ctypes.c_void_p)"
                )
                code.append(f"{_cb_last_call_args}[{sdfg_arg_index}].value = {arg}.data_ptr()")
                code.append(f"assert gtx_common.Domain.is_finite({arg}.domain)")
                for i, array_size in enumerate(sdfg_arg_desc.shape):
                    if (
                        isinstance(array_size, dace.symbolic.SymbolicType)
                        and not array_size.is_constant()
                    ):
                        dim_range = f"{arg}.domain.ranges[{i}]"
                        rstart = gtx_dace_utils.range_start_symbol(param_name, i)
                        rstop = gtx_dace_utils.range_stop_symbol(param_name, i)
                        for suffix, symbol_name in [("start", rstart), ("stop", rstop)]:
                            value = f"{dim_range}.{suffix}"
                            _parse_gt_param(
                                symbol_name,
                                FIELD_SYMBOL_GT_TYPE,
                                value,
                                code,
                                sdfg_arglist,
                                sdfg_argnames,
                                make_persistent,
                            )
                    else:
                        # the array shape is set to constant value
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].shape[i] == {arg}.ndarray.shape[i]"
                        )
                for i, array_stride in enumerate(sdfg_arg_desc.strides):
                    if (
                        isinstance(array_stride, dace.symbolic.SymbolicType)
                        and not array_stride.is_constant()
                    ):
                        assert array_stride.name == gtx_dace_utils.field_stride_symbol_name(
                            param_name, i
                        )
                        value = f"{_cb_get_stride}({arg}.ndarray, {i})"
                        _parse_gt_param(
                            array_stride.name,
                            FIELD_SYMBOL_GT_TYPE,
                            value,
                            code,
                            sdfg_arglist,
                            sdfg_argnames,
                            make_persistent,
                        )
                    else:
                        # the array stride is set to constant value
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].strides[i] == stride"
                        )

        elif isinstance(param_type, ts.ScalarType):
            assert isinstance(sdfg_arg_desc, dace.data.Scalar)
            if gtx_dace_utils.is_field_symbol(param_name) and make_persistent:
                # only emit some debug code
                _validate_sdfg_scalar_arg(code, arg, sdfg_arg_desc, sdfg_arg_index)
            else:
                _update_sdfg_scalar_arg(code, arg, sdfg_arg_desc, sdfg_arg_index)

        else:
            raise ValueError(f"Unexpected paramter type {param_type}")


def create_sdfg_bindings(
    program_source: stages.ProgramSource[SrcL, languages.LanguageWithHeaderFilesSettings],
    bind_func_name: str,
    make_persistent: bool,
) -> stages.BindingSource[SrcL, languages.Python]:
    """
    Creates the Python bindings to call an SDFG with a GT4Py arguments list.

    Args:
        program_source: The json representation of the SDFG.
        make_persistent: When True, it is safe to assume that the field layout does
            not change across mutiple program calls.

    Returns:
        The Python code to convert call arguments from gt4py canonical form to the
        SDFG canonical form.
    """
    sdfg = dace.SDFG.from_json(program_source.source_code)

    # `dace.SDFG.arglist()` returns an ordered dictionary that maps the argument
    # name to its data type, in the same order as arguments appear in the program ABI.
    # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
    sdfg_arglist = sdfg.arglist()
    sdfg_argnames = [arg_name for arg_name in sdfg_arglist.keys()]

    code = codegen.TextBlock()

    code.append("import ctypes")
    code.append("from gt4py.next import common as gtx_common, field_utils")
    code.empty_line()
    code.append(f"""\
def {_cb_get_stride}(ndarray, dim_index):
    stride, remainder = divmod(ndarray.strides[dim_index], ndarray.itemsize)
    assert remainder == 0
    return stride
""")
    code.empty_line()
    code.append(
        "def {funname}({arg0}, {arg1}, {arg2}, {arg3}):".format(
            funname=bind_func_name,
            arg0=_cb_device,
            arg1=_cb_sdfg_argtypes,
            arg2=_cb_args,
            arg3=_cb_last_call_args,
        )
    )
    code.indent()
    for i, param in enumerate(program_source.entry_point.parameters):
        arg = f"{_cb_args}[{i}]"
        assert isinstance(param.type_, ts.DataType)
        _parse_gt_param(
            param.name, param.type_, arg, code, sdfg_arglist, sdfg_argnames, make_persistent
        )
    code.dedent()

    src = codegen.format_python_source(code.text)
    return stages.BindingSource(src, library_deps=tuple())


def bind_sdfg(
    inp: stages.ProgramSource[SrcL, languages.LanguageWithHeaderFilesSettings],
    bind_func_name: str,
    make_persistent: bool,
) -> stages.CompilableSource[SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python]:
    """
    Method to be used as workflow stage for generation of SDFG bindings.

    Refer to `create_bindings` documentation.
    """
    return stages.CompilableSource(
        program_source=inp,
        binding_source=create_sdfg_bindings(inp, bind_func_name, make_persistent),
    )
