# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TypeVar

import dace

from gt4py.eve import codegen
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils
from gt4py.next.program_processors.runners.dace.workflow import factory as dace_workflow_factory
from gt4py.next.type_system import type_specifications as ts


SrcL = TypeVar("SrcL", bound=languages.SDFG, covariant=True)

_cb_device = "device"
_cb_sdfg_argtypes = "sdfg_argtypes"
_cb_args = "args"
_cb_last_call_args = "last_call_args"


def _update_sdfg_scalar_arg(
    code: codegen.TextBlock, rhs: str, sdfg_arg_desc: dace.data.Data, sdfg_arg_index: int
) -> None:
    assert isinstance(sdfg_arg_desc, dace.data.Scalar)
    code.append(f"assert isinstance({_cb_last_call_args}[{sdfg_arg_index}], ctypes._SimpleCData)")
    code.append(f"actype = {_cb_sdfg_argtypes}[{sdfg_arg_index}].dtype.as_ctypes()")
    code.append(f"{_cb_last_call_args}[{sdfg_arg_index}] = actype({rhs})")


def _parse_gt_param(
    param_name: str,
    param_type: ts.DataType,
    arg: str,
    code: codegen.TextBlock,
    sdfg_arglist: dict[str, dace.data.Data],
    sdfg_argnames: list[str],
    make_persistent: bool,
) -> None:
    if isinstance(param_type, ts.TupleType):
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
        return

    if param_name not in sdfg_arglist:
        # symbol not used in the SDFG
        assert isinstance(param_type, ts.ScalarType)
        assert gtx_dace_utils.is_field_symbol(param_name)
        return

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
            if not make_persistent:
                code.append(f"domain = {arg}.domain")
                code.append(f"ndarray = {arg}.ndarray")
                code.append("assert gtx_common.Domain.is_finite(domain)")
                for i, size in enumerate(sdfg_arg_desc.shape):
                    if isinstance(size, dace.symbolic.SymbolicType):
                        if gtx_dace_utils.is_connectivity_identifier(param_name):
                            value = f"ndarray.shape[{i}]"
                            symbol_name = gtx_dace_utils.field_size_symbol_name(param_name, i)
                            sdfg_symbol_arg_desc = sdfg_arglist[symbol_name]
                            sdfg_symbol_arg_index = sdfg_argnames.index(symbol_name)
                            _update_sdfg_scalar_arg(
                                code, value, sdfg_symbol_arg_desc, sdfg_symbol_arg_index
                            )
                        else:
                            dim_range = f"domain.ranges[{i}]"
                            r0_symbol = gtx_dace_utils.range_start_symbol(param_name, i)
                            r1_symbol = gtx_dace_utils.range_stop_symbol(param_name, i)
                            for range_suffix, symbol_name in [
                                ("start", r0_symbol),
                                ("stop", r1_symbol),
                            ]:
                                value = f"{dim_range}.{range_suffix}"
                                if symbol_name in sdfg_arglist:
                                    # symbols that are not used are removed from the SDFG arglist
                                    sdfg_symbol_arg_desc = sdfg_arglist[symbol_name]
                                    sdfg_symbol_arg_index = sdfg_argnames.index(symbol_name)
                                    _update_sdfg_scalar_arg(
                                        code, value, sdfg_symbol_arg_desc, sdfg_symbol_arg_index
                                    )
                    else:
                        # the array shape is set to constant value
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].shape[i] == ndarray.shape[i]"
                        )
                for i, stride in enumerate(sdfg_arg_desc.strides):
                    if isinstance(stride, dace.symbolic.SymbolicType):
                        value = f"ndarray.strides[{i}]"
                        symbol_name = gtx_dace_utils.field_stride_symbol_name(param_name, i)
                        if symbol_name in sdfg_arglist:
                            # symbols that are not used are removed from the SDFG arglist
                            sdfg_symbol_arg_desc = sdfg_arglist[symbol_name]
                            sdfg_symbol_arg_index = sdfg_argnames.index(symbol_name)
                            code.append(f"stride, remainder = divmod({value}, ndarray.itemsize)")
                            code.append("assert remainder == 0")
                            _update_sdfg_scalar_arg(
                                code, "stride", sdfg_symbol_arg_desc, sdfg_symbol_arg_index
                            )
                    else:
                        # the array stride is set to constant value
                        code.append(
                            f"assert {_cb_sdfg_argtypes}[{sdfg_arg_index}].strides[i] == stride"
                        )

    elif isinstance(param_type, ts.ScalarType):
        assert isinstance(sdfg_arg_desc, dace.data.Scalar)
        _update_sdfg_scalar_arg(code, arg, sdfg_arg_desc, sdfg_arg_index)

    else:
        raise ValueError(f"Unexpected paramter type {param_type}")


def create_bindings(
    program_source: stages.ProgramSource[SrcL, languages.LanguageWithHeaderFilesSettings],
    make_persistent: bool,
) -> stages.BindingSource[SrcL, languages.Python]:
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
    code.append(
        "def {funname}({arg0}, {arg1}, {arg2}, {arg3}):".format(
            funname=dace_workflow_factory.GT_DACE_BINDING_FUNCTION_NAME,
            arg0=_cb_device,
            arg1=_cb_sdfg_argtypes,
            arg2=_cb_args,
            arg3=_cb_last_call_args,
        )
    )
    code.indent()
    for i, param in enumerate(program_source.entry_point.parameters):
        code.empty_line()
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
    make_persistent: bool,
) -> stages.CompilableSource[SrcL, languages.LanguageWithHeaderFilesSettings, languages.Python]:
    return stages.CompilableSource(
        program_source=inp, binding_source=create_bindings(inp, make_persistent)
    )
