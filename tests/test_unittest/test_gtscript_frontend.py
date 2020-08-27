# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import textwrap
import types

import numpy as np
import pytest

from gt4py import definitions as gt_definitions
from gt4py import gtscript
from gt4py import utils as gt_utils
from gt4py.frontend import gtscript_frontend as gt_frontend

from ..definitions import id_version


# ---- Utilities -----
frontend = gt_frontend.GTScriptFrontend


def compile_definition(
    definition_func,
    name: str,
    module: str,
    *,
    externals: dict = None,
    dtypes: dict = None,
    rebuild=False,
    **kwargs,
):
    gtscript._set_arg_dtypes(definition_func, dtypes=dtypes or {})
    build_options = gt_definitions.BuildOptions(
        name=name, module=module, rebuild=rebuild, backend_opts=kwargs, build_info=None
    )

    options_id = gt_utils.shashed_id(build_options)
    stencil_id = frontend.get_stencil_id(
        build_options.qualified_name, definition_func, externals, options_id
    )
    definition_ir = gt_frontend.GTScriptParser(
        definition_func, externals=externals or {}, options=build_options
    ).run()

    return stencil_id, definition_ir


# ---- Tests-----

GLOBAL_BOOL_CONSTANT = True
GLOBAL_CONSTANT = 1.0
GLOBAL_NESTED_CONSTANTS = types.SimpleNamespace(A=100, B=200)
GLOBAL_VERY_NESTED_CONSTANTS = types.SimpleNamespace(nested=types.SimpleNamespace(A=1000, B=2000))


@gtscript.function
def add_external_const(a):
    return a + 10.0 + GLOBAL_CONSTANT


@gtscript.function
def identity(field_in):
    return field_in


@gtscript.function
def sinus(field_in):
    return sin(field_in)


class TestInlinedExternals:
    def test_all_legal_combinations(self, id_version):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = (
                    (
                        inout_field[0, 0, 0]
                        + GLOBAL_CONSTANT
                        + GLOBAL_NESTED_CONSTANTS.A
                        + GLOBAL_VERY_NESTED_CONSTANTS.nested.A
                    )
                    if GLOBAL_BOOL_CONSTANT
                    else 0
                )

        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    def test_missing(self, id_version):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptSymbolError, match=r".*MISSING_CONSTANT.*"):
            compile_definition(definition_func, "test_missing_symbol", module, externals=externals)

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + GLOBAL_NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError, match=r".*GLOBAL_NESTED_CONSTANTS.missing.*"
        ):
            compile_definition(
                definition_func, "test_missing_nested_symbol", module, externals=externals
            )

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, id_version, value_type):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        WRONG_VALUE_CONSTANT = value_type()

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        with pytest.raises(gt_frontend.GTScriptSymbolError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestImportedExternals:
    def test_all_legal_combinations(self, id_version):
        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(
            BOOL_CONSTANT=-1.0,
            CONSTANT=-2.0,
            NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200),
            VERY_NESTED_CONSTANTS=types.SimpleNamespace(
                nested=types.SimpleNamespace(A=-1000, B=-2000)
            ),
        )

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__externals__ import (
                BOOL_CONSTANT,
                CONSTANT,
                NESTED_CONSTANTS,
                VERY_NESTED_CONSTANTS,
            )
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = (
                    (
                        inout_field[0, 0, 0]
                        + CONSTANT
                        + NESTED_CONSTANTS.A
                        + VERY_NESTED_CONSTANTS.nested.A
                    )
                    if GLOBAL_BOOL_CONSTANT
                    else 0
                )

        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    def test_missing(self, id_version):
        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(CONSTANT=-2.0, NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200))

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__externals__ import MISSING_CONSTANT
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*MISSING_CONSTANT.*"):
            compile_definition(definition_func, "test_missing_symbol", module, externals=externals)

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__externals__ import NESTED_CONSTANTS
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError, match=r".*NESTED_CONSTANTS.missing.*"
        ):
            compile_definition(
                definition_func, "test_missing_nested_symbol", module, externals=externals
            )

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, id_version, value_type):
        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__externals__ import WRONG_VALUE_CONSTANT
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(WRONG_VALUE_CONSTANT=value_type())

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestExternalsWithSubroutines:
    def test_all_legal_combinations(self, id_version):
        @gtscript.function
        def _stage_laplacian_x(dx, phi):
            lap = add_external_const(phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx)
            return lap

        @gtscript.function
        def _stage_laplacian_y(dy, phi):
            lap = (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dy * dy)
            return lap

        @gtscript.function
        def _stage_laplacian(dx, dy, phi):
            from __externals__ import stage_laplacian_x, stage_laplacian_y

            lap_x = stage_laplacian_x(dx=dx, phi=phi)
            lap_y = stage_laplacian_y(dy=dy, phi=phi)
            lap = lap_x[0, 0, 0] + lap_y[0, 0, 0]

            return lap

        def definition_func(
            in_phi: gtscript.Field[np.float64],
            in_gamma: gtscript.Field[np.float64],
            out_phi: gtscript.Field[np.float64],
            out_field: gtscript.Field[np.float64],
            *,
            dx: float,
            dy: float,
        ):
            from gt4py.__externals__ import stage_laplacian, stage_laplacian_x, stage_laplacian_y
            from gt4py.__gtscript__ import BACKWARD, FORWARD, PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                lap = stage_laplacian(dx=dx, dy=dy, phi=in_phi) + GLOBAL_CONSTANT
                out_phi = in_gamma[0, 0, 0] * lap[0, 0, 0]

            with computation(PARALLEL), interval(...):
                tmp_out = identity(in_phi)
                out_phi = tmp_out + 1

            with computation(PARALLEL), interval(...):
                tmp_out2 = identity(in_gamma)
                out_field = out_phi + tmp_out2

        module = f"TestExternalsWithSubroutines_test_module_{id_version}"
        externals = {
            "stage_laplacian": _stage_laplacian,
            "stage_laplacian_x": _stage_laplacian_x,
            "stage_laplacian_y": _stage_laplacian_y,
        }
        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )


class TestImports:
    def test_all_legal_combinations(self, id_version):
        def definition_func(inout_field: gtscript.Field[float]):
            from __externals__ import EXTERNAL
            from __gtscript__ import BACKWARD, FORWARD, PARALLEL, computation, interval
            from gt4py.__externals__ import EXTERNAL
            from gt4py.__gtscript__ import BACKWARD, FORWARD, PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        module = f"TestImports_test_module_{id_version}"
        externals = dict(EXTERNAL=1.0)
        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    @pytest.mark.parametrize(
        "id_case,import_line",
        list(
            enumerate(
                [
                    "import gt4py",
                    "from externals import EXTERNAL",
                    "from gt4py import __gtscript__",
                    "from gt4py import __externals__",
                    "from gt4py.gtscript import computation",
                    "from gt4py.externals import EXTERNAL",
                ]
            )
        ),
    )
    def test_wrong_imports(self, id_case, import_line, id_version):
        module = f"TestImports_test_module_{id_version}"
        externals = {}

        definition_source = textwrap.dedent(
            f"""
        def definition_func(inout_field: gtscript.Field[float]):
            {import_line}

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0]
"""
        )

        context = dict(gtscript=gtscript)
        exec(definition_source, context)
        definition_func = context["definition_func"]
        definition_func.__exec_source__ = definition_source

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            compile_definition(
                definition_func, f"test_wrong_imports_{id_case}", module, externals=externals
            )


class TestDTypes:
    @pytest.mark.parametrize(
        "id_case,test_dtype",
        list(enumerate([bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64])),
    )
    def test_all_legal_dtypes(self, id_case, test_dtype, id_version):
        def definition_func(
            in_field: gtscript.Field[test_dtype],
            out_field: gtscript.Field[test_dtype],
            param: test_dtype,
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"
        compile_definition(definition_func, "test_all_legal_dtypes", module)

        def definition_func(
            in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"], param: "dtype"
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"
        compile_definition(
            definition_func, "test_all_legal_dtypes", module, dtypes={"dtype": test_dtype}
        )

    @pytest.mark.parametrize(
        "id_case,test_dtype", list(enumerate([str, np.uint32, np.uint64, dict, map, bytes]))
    )
    def test_invalid_inlined_dtypes(self, id_case, test_dtype, id_version):
        with pytest.raises(ValueError, match=r".*data type descriptor.*"):

            def definition_func(
                in_field: gtscript.Field[test_dtype],
                out_field: gtscript.Field[test_dtype],
                param: test_dtype,
            ):
                with computation(PARALLEL), interval(...):
                    out_field = in_field + param

    @pytest.mark.parametrize(
        "id_case,test_dtype", list(enumerate([str, np.uint32, np.uint64, dict, map, bytes]))
    )
    def test_invalid_external_dtypes(self, id_case, test_dtype, id_version):
        def definition_func(
            in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"], param: "dtype"
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"

        with pytest.raises(ValueError, match=r".*data type descriptor.*"):
            compile_definition(
                definition_func,
                "test_invalid_external_dtypes",
                module,
                dtypes={"dtype": test_dtype},
            )


class TestAssignmentSyntax:
    def test_ellipsis(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                out_field[...] = in_field

    def test_offset(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                out_field[0, 0, 0] = in_field

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field[0, 0, 1] = in_field

        @gtscript.stencil(backend="debug", externals={"offset": 0})
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            from gt4py.__externals__ import offset

            with computation(PARALLEL), interval(...):
                out_field[0, 0, offset] = in_field

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug", externals={"offset": 1})
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                from gt4py.__externals__ import offset

                with computation(PARALLEL), interval(...):
                    out_field[0, 0, offset] = in_field

    def test_slice(self):

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field[:, :, :] = in_field

    def test_string(self):
        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field["a_key"] = in_field

    def test_temporary(self):
        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="No subscript allowed in assignment to temporaries",
        ):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    tmp[...] = in_field
                    out_field = tmp

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="No subscript allowed in assignment to temporaries",
        ):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    tmp[0, 0, 0] = 2 * in_field
                    out_field = tmp

    def test_augmented(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += 2.0
                in_field -= 0.5
                in_field /= 0.5
                in_field *= 4.0


class TestNestedWithSyntax:
    def test_nested_with(self):
        @gtscript.stencil(backend="debug")
        def definition(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            with computation(PARALLEL):
                with interval(...):
                    in_field = out_field

    def test_nested_with_reordering(self):
        def definition_fw(
            in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]
        ):
            from gt4py.__gtscript__ import FORWARD, computation, interval

            with computation(FORWARD):
                with interval(1, 2):
                    in_field = out_field + 1
                with interval(0, 1):
                    in_field = out_field + 2

        def definition_bw(
            in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]
        ):
            from gt4py.__gtscript__ import FORWARD, computation, interval

            with computation(BACKWARD):
                with interval(1, 2):
                    in_field = out_field + 1
                with interval(0, 1):
                    in_field = out_field + 2

        definitions = [
            # name, expected axis bounds, definition
            ("fw", [(0, 1), (1, 2)], definition_fw),
            ("bw", [(1, 2), (0, 1)], definition_bw),
        ]

        for name, axis_bounds, definition in definitions:
            # generate DIR
            _, definition_ir = compile_definition(
                definition,
                f"test_nested_with_reordering_{name}",
                f"TestImports_test_module_{id_version}",
            )

            # test for correct ordering
            for i, axis_bound in enumerate(axis_bounds):
                interval = definition_ir.computations[i].interval
                assert interval.start.offset == axis_bound[0]
                assert interval.end.offset == axis_bound[1]


class TestNativeFunctions:
    def test_simple_call(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += sin(in_field)

    def test_offset_arg(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += sin(in_field[1, 0, 0])

    def test_nested_calls(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += sin(abs(in_field))

    def test_nested_external_call(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += sin(add_external_const(in_field))

    def test_multi_nested_calls(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += min(abs(sin(add_external_const(in_field))), -0.5)

    def test_native_in_function(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field += sinus(in_field)

    def test_native_function_unary(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field = not isfinite(in_field)

    def test_native_function_binary(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field = asin(in_field) + 1

    def test_native_function_ternary(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                in_field = asin(in_field) + 1 if 1 < in_field else sin(in_field)


class TestRegions:
    def test_on_interval_only(self):
        module = f"TestRegion_on_interval_only_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0, ie

            with computation(PARALLEL), interval(...), parallel(region[i0 : 1 + ie, :]):
                in_f = 1.0

        stencil_id, def_ir = compile_definition(stencil, "stencil", module, externals=externals)

        assert len(def_ir.computations) == 1
        assert def_ir.computations[0].parallel_interval is not None

    def test_single_on_interval_only(self):
        module = f"TestRegion_single_on_interval_only_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0, ie

            with computation(PARALLEL), interval(...), parallel(region[i0, :]):
                in_f = 1.0

        stencil_id, def_ir = compile_definition(stencil, "stencil", module, externals=externals)

        assert len(def_ir.computations) == 1
        assert def_ir.computations[0].parallel_interval is not None

    def test_with_default(self):
        module = f"TestRegion_with_default_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0, ie, j0, je

            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with parallel(region[i0 : 1 + ie, :], region[:, j0 : 1 + je]):
                    in_f = 1.0

        stencil_id, def_ir = compile_definition(stencil, "stencil", module, externals=externals)

        assert len(def_ir.computations) == 3
        assert def_ir.computations[0].parallel_interval is None
        assert def_ir.computations[1].parallel_interval is not None
        assert def_ir.computations[2].parallel_interval is not None

    def test_multiple_with_default(self):
        module = f"TestRegion_multiple_with_default_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0, ie, j0, je

            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with parallel(region[i0 : 1 + ie, :]):
                    in_f = 1.0
                with parallel(region[:, j0 : 1 + je]):
                    in_f = 2.0

        stencil_id, def_ir = compile_definition(stencil, "stencil", module, externals=externals)

        assert len(def_ir.computations) == 3
        assert def_ir.computations[0].parallel_interval is None
        assert def_ir.computations[1].parallel_interval is not None
        assert def_ir.computations[2].parallel_interval is not None

    def test_error_undefined(self):
        module = f"TestRegion_error_undefined_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0  # forget to add 'ia'

            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with parallel(region[i0 : 1 + ia, :]):
                    in_f = 1.0

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Unknown symbol"):
            compile_definition(stencil, "stencil", module, externals=externals)

    def test_error_nested(self):
        module = f"TestRegion_error_nested_{id_version}"
        externals = {}

        def stencil(in_f: gtscript.Field[np.float_]):
            from __splitters__ import i0, ie, j0, je

            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with parallel(region[i0 : 1 + ie, :]):
                    in_f = 1.0
                    with parallel(region[:, j0 : 1 + je]):
                        in_f = 2.0

        # TODO This error could be improved
        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="Invalid 'computation' specification"
        ):
            compile_definition(stencil, "stencil", module, externals=externals)
