# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final

import dace

from gt4py.eve import codegen
from gt4py.next.iterator import builtins as itir_builtins
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils
from gt4py.next.type_system import type_specifications as ts


FIELD_SYMBOL_GT_TYPE: Final[ts.ScalarType] = ts.ScalarType(
    kind=getattr(ts.ScalarKind, itir_builtins.INTEGER_INDEX_BUILTIN.upper())
)

_cb_args: Final[str] = "args"
_cb_device: Final[str] = "device"
_cb_sdfg_argtypes: Final[str] = "sdfg_argtypes"
_cb_sdfg_call_args: Final[str] = "sdfg_call_args"


def _update_sdfg_scalar_arg(
    code: codegen.TextBlock,
    sdfg_arg_desc: dace.data.Data,
    sdfg_arg_index: int,
    call_arg: str,
) -> None:
    """
    Emit Python code to update a scalar argument in the SDFG arglist
    with the argument value passed to the gt4py program call.
    """
    assert isinstance(sdfg_arg_desc, dace.data.Scalar)
    actype = sdfg_arg_desc.dtype.as_ctypes()
    actype_call = f"{actype.__module__}.{actype.__name__}"
    code.append(f"assert isinstance({_cb_sdfg_call_args}[{sdfg_arg_index}], ctypes._SimpleCData)")
    code.append(f"{_cb_sdfg_call_args}[{sdfg_arg_index}] = {actype_call}({call_arg})")


def _unpack_args(code: codegen.TextBlock, num_args: int, arg_name: str) -> list[str]:
    """Unpack a sequence of arguments (either a list or a tuple) into variables.

    Each element of the given sequence gets a name 'arg_name' with an index-based suffix.

    >>> code = codegen.TextBlock()
    >>> _unpack_args(code, 3, "var")
    ['var_0', 'var_1', 'var_2']
    >>> code.lines
    ['var_0, var_1, var_2, = var']
    >>> _unpack_args(code, 1, "var_2")
    ['var_2_0']
    >>> code.lines
    ['var_0, var_1, var_2, = var', 'var_2_0, = var_2']
    """
    tuple_args = [f"{arg_name}_{i}" for i in range(num_args)]
    if num_args == 0:
        raise ValueError("Cannot unpack argument with length zero.")
    else:
        # The trailing comma is needed to unpack single-element tuples
        code.append(f"{', '.join(tuple_args)}, = {arg_name}")
    return tuple_args


def _parse_gt_param(
    param_name: str,
    param_type: ts.DataType,
    arg: str,
    code: codegen.TextBlock,
    sdfg_arglist: dict[str, dace.data.Data],
) -> None:
    """Emit Python code to parse a program argument and set the required fields in the SDFG arglist.

    For scalar arguments, a single field is set in the SDFG arglist.

    For array arguments, in addition to the data pointer, the fields for array shape
    and strides are also set in SDFG arglist. This results in nested calls to
    `_parse_gt_param()` with the scalar values of array shape and strides.

    For tuple arguments, this function is recursively called on all elements of the tuple.
    """
    if isinstance(param_type, ts.TupleType):
        # Each element of a tuple gets a name with an index-based suffix and it is recursively visited.
        tuple_args = _unpack_args(code=code, num_args=len(param_type.types), arg_name=arg)
        for i, (tuple_arg, tuple_arg_type) in enumerate(zip(tuple_args, param_type.types)):
            assert isinstance(tuple_arg_type, ts.DataType)
            _parse_gt_param(
                param_name=f"{param_name}_{i}",
                param_type=tuple_arg_type,
                arg=tuple_arg,
                code=code,
                sdfg_arglist=sdfg_arglist,
            )

    elif param_name not in sdfg_arglist:
        # There are two reasons for this case:
        #   1) The argument is a symbol/scalar that is not used in the generated code.
        #   2) The argument was demoted, see `demote_fields` argument of `gt_auto_optimize()`
        #       and was not put back.
        pass

    else:
        sdfg_arg_desc = sdfg_arglist[param_name]
        sdfg_arg_index = list(sdfg_arglist.keys()).index(param_name)

        if isinstance(param_type, ts.FieldType):
            if len(param_type.dims) == 0:
                # Pass zero-dimensional fields as scalars.
                assert isinstance(sdfg_arg_desc, dace.data.Scalar)
                _update_sdfg_scalar_arg(
                    code=code,
                    sdfg_arg_desc=sdfg_arg_desc,
                    sdfg_arg_index=sdfg_arg_index,
                    call_arg=f"{arg}.as_scalar()",
                )
            else:
                assert isinstance(sdfg_arg_desc, dace.data.Array)
                code.append(f"assert field_utils.verify_device_field_type({arg}, {_cb_device})")
                code.append(
                    f"assert isinstance({_cb_sdfg_call_args}[{sdfg_arg_index}], ctypes.c_void_p)"
                )
                code.append(f"{arg}_buffer_info = {arg}.__gt_buffer_info__")

                code.append(
                    f"{_cb_sdfg_call_args}[{sdfg_arg_index}].value = {arg}_buffer_info.data_ptr"
                )
                for i, (dim, array_size) in enumerate(
                    zip(param_type.dims, sdfg_arg_desc.shape, strict=True)
                ):
                    if isinstance(array_size, int) or str(array_size).isdigit():
                        # The array shape in this dimension is set at compile-time.
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].shape[{i}] == {arg}_buffer_info.shape[{i}]"
                        )
                    else:
                        # The array shape is defined as a sequence of expressions
                        # like 'range_stop - range_start', where 'range_start' and
                        # 'range_stop' are the SDFG symbols for the domain range.
                        arg_range = f"{arg}.domain.ranges[{i}]"
                        rstart = gtx_dace_utils.range_start_symbol(param_name, dim)
                        rstop = gtx_dace_utils.range_stop_symbol(param_name, dim)
                        for suffix, symbol_name in [("start", rstart), ("stop", rstop)]:
                            _parse_gt_param(
                                param_name=symbol_name,
                                param_type=FIELD_SYMBOL_GT_TYPE,
                                arg=f"{arg_range}.{suffix}",
                                code=code,
                                sdfg_arglist=sdfg_arglist,
                            )
                for i, array_stride in enumerate(sdfg_arg_desc.strides):
                    arg_stride = f"{arg}_buffer_info.elem_strides[{i}]"
                    if isinstance(array_stride, int) or str(array_stride).isdigit():
                        # The array stride is set to constant value in this dimension.
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].strides[{i}] == {arg_stride}"
                        )
                    else:
                        symbol_name = gtx_dace_utils.field_stride_symbol_name(param_name, i)
                        assert str(array_stride) == symbol_name
                        # The strides of a global array are defined by a sequence
                        # of SDFG symbols.
                        _parse_gt_param(
                            param_name=symbol_name,
                            param_type=FIELD_SYMBOL_GT_TYPE,
                            arg=arg_stride,
                            code=code,
                            sdfg_arglist=sdfg_arglist,
                        )

        elif isinstance(param_type, ts.ScalarType):
            assert isinstance(sdfg_arg_desc, dace.data.Scalar)
            _update_sdfg_scalar_arg(
                code=code,
                sdfg_arg_desc=sdfg_arg_desc,
                sdfg_arg_index=sdfg_arg_index,
                call_arg=arg,
            )

        else:
            raise ValueError(f"Unexpected paramter type {param_type}")


def _create_sdfg_bindings(
    program_source: stages.ProgramSource[languages.SDFG, languages.LanguageSettings],
    bind_func_name: str,
) -> stages.BindingSource[languages.SDFG, languages.Python]:
    """
    Creates a Python translation function to convert the GT4Py arguments list
    to the SDFG calling convention.

    Args:
        program_source: The json representation of the SDFG.
        bind_func_name: Name to use for the translation function.

    Returns:
        The Python code to convert call arguments from gt4py canonical form to the
        SDFG canonical form.
    """
    sdfg = dace.SDFG.from_json(program_source.source_code)

    # `dace.SDFG.arglist()` returns an ordered dictionary that maps the argument
    # name to its data type, in the same order as arguments appear in the program ABI.
    # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
    sdfg_arglist = sdfg.arglist()

    code = codegen.TextBlock()

    code.append("import ctypes")
    code.append("from gt4py.next import common as gtx_common, field_utils")
    code.empty_line()
    code.append(
        "def {funname}({arg0}, {arg1}, {arg2}, {arg3}):".format(
            funname=bind_func_name,
            arg0=_cb_device,
            arg1=_cb_sdfg_argtypes,
            arg2=_cb_args,
            arg3=_cb_sdfg_call_args,
        )
    )

    # The SDFG binding function is used with fast-call, to update the SDFG arguments
    #   list, therefore it is only used from the second time the SDFG is called.
    #   On the first time, we use the regular SDFG call, which constructs the SDFG
    #   arguments list and validates that all data containers and free symbols are set.
    with code.indented():
        arg_vars = _unpack_args(
            code=code, num_args=len(program_source.entry_point.parameters), arg_name=_cb_args
        )
        for param, arg in zip(program_source.entry_point.parameters, arg_vars):
            assert isinstance(param.type_, ts.DataType)
            _parse_gt_param(param.name, param.type_, arg, code, sdfg_arglist)

    src = codegen.format_python_source(code.text)
    return stages.BindingSource(src, library_deps=tuple())


def bind_sdfg(
    inp: stages.ProgramSource[languages.SDFG, languages.LanguageSettings],
    bind_func_name: str,
) -> stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python]:
    """
    Method to be used as workflow stage for generation of SDFG bindings.

    Refer to `_create_sdfg_bindings` documentation.
    """
    return stages.CompilableSource(
        program_source=inp,
        binding_source=_create_sdfg_bindings(inp, bind_func_name),
    )
