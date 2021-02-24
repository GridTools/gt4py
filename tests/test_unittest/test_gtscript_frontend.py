# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import gt4py.definitions as gt_definitions
import gt4py.ir as gt_ir
import gt4py.utils as gt_utils
from gt4py import gtscript
from gt4py.frontend import gtscript_frontend as gt_frontend
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

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
    _, original_annotations = gtscript._set_arg_dtypes(definition_func, dtypes=dtypes or {})
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

    setattr(definition_func, "__annotations__", original_annotations)

    return stencil_id, definition_ir


# ---- Tests-----

GLOBAL_BOOL_CONSTANT = True
GLOBAL_CONSTANT = 1.0
GLOBAL_NESTED_CONSTANTS = types.SimpleNamespace(A=100, B=200)
GLOBAL_VERY_NESTED_CONSTANTS = types.SimpleNamespace(nested=types.SimpleNamespace(A=1000, B=2000))


@gtscript.function
def add_external_const(a):
    return a + 10.0 + GLOBAL_CONSTANT


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

    def test_recursive_function_imports(self, id_version):
        module = f"TestInlinedExternals_test_recursive_imports_{id_version}"

        @gtscript.function
        def func_deeply_nested():
            from __externals__ import another_const

            return another_const

        @gtscript.function
        def func_nested():
            from __externals__ import const

            return const + func_deeply_nested()

        @gtscript.function
        def func():
            from __externals__ import other_call

            return other_call()

        def definition_func(inout_field: gtscript.Field[float]):
            from __externals__ import some_call

            with computation(PARALLEL), interval(...):
                inout_field = func() + some_call()

        stencil_id, def_ir = compile_definition(
            definition_func,
            "test_recursive_imports",
            module,
            externals={
                "some_call": func,
                "other_call": func_nested,
                "const": GLOBAL_CONSTANT,
                "another_const": GLOBAL_CONSTANT,
            },
        )
        assert set(def_ir.externals.keys()) == {
            "some_call",
            "const",
            "other_call",
            "func",
            "another_const",
            "tests.test_unittest.test_gtscript_frontend.func_nested.func_deeply_nested",
        }

    def test_decorated_freeze(self):
        A = 0

        @gtscript.function
        def some_function():
            return A

        module = f"TestInlinedExternals_test_undecorated_delay_{id_version}"
        externals = {"func": some_function}

        A = 1

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = some_function()

        stencil_id, def_ir = compile_definition(
            definition_func, "test_decorated_freeze", module, externals=externals
        )

        stmt = def_ir.computations[0].body.stmts[0]
        assert isinstance(stmt.value, gt_ir.ScalarLiteral) and stmt.value.value == 0

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, id_version, value_type):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        WRONG_VALUE_CONSTANT = value_type()

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestFunction:
    def test_error_invalid(self, id_version):
        module = f"TestFunction_test_module_{id_version}"
        externals = {}

        def func():
            return 1.0

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func()

        with pytest.raises(TypeError, match=r"func is not a gtscript function"):
            compile_definition(definition_func, "test_error_invalid", module, externals=externals)


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

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(WRONG_VALUE_CONSTANT=value_type())

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestIntervalSyntax:
    def test_simple(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(0, 1):
                field = 0

        module = f"TestIntervalSyntax_simple_{id_version}"
        externals = {}
        stencil_id, def_ir = compile_definition(
            definition_func, "test_simple", module, externals=externals
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.START, offset=0, loc=loc
        )
        assert def_ir.computations[0].interval.end == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.START, offset=1, loc=loc
        )

    def test_none(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(1, None):
                field = 0

        module = f"TestIntervalSyntax_none_{id_version}"
        externals = {}
        stencil_id, def_ir = compile_definition(
            definition_func, "test_none", module, externals=externals
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.START, offset=1, loc=loc
        )
        assert def_ir.computations[0].interval.end == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.END, offset=0, loc=loc
        )

    def test_externals(self):
        def definition_func(field: gtscript.Field[float]):
            from __externals__ import kstart

            with computation(PARALLEL), interval(kstart, -1):
                field = 0

        module = f"TestIntervalSyntax_externals_{id_version}"
        for kstart in (3, gtscript.K[3]):
            # An implementation quirk allows us to use gtscript.K[3] here,
            # although it is not great form to do so, since two-argument syntax
            # should not use AxisOffsets.
            externals = {"kstart": kstart}
            stencil_id, def_ir = compile_definition(
                definition_func, "test_externals", module, externals=externals
            )
            loc = def_ir.computations[0].interval.loc
            assert def_ir.computations[0].interval.start == gt_ir.AxisBound(
                level=gt_ir.LevelMarker.START, offset=3, loc=loc
            )
            assert def_ir.computations[0].interval.end == gt_ir.AxisBound(
                level=gt_ir.LevelMarker.END, offset=-1, loc=loc
            )

    def test_axisinterval(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(K[1:-1]):
                field = 0

        module = f"TestIntervalSyntax_simple_{id_version}"
        externals = {}
        stencil_id, def_ir = compile_definition(
            definition_func, "test_externals", module, externals=externals
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.START, offset=1, loc=loc
        )
        assert def_ir.computations[0].interval.end == gt_ir.AxisBound(
            level=gt_ir.LevelMarker.END, offset=-1, loc=loc
        )

    def test_error_none(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(None, -1):
                field = 0

        module = f"TestIntervalSyntax_error_none_{id_version}"
        externals = {}

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="Invalid interval range specification"
        ):
            compile_definition(definition_func, "test_error_none", module, externals=externals)

    def test_error_do_not_mix(self):
        def definition_func(field: gtscript.Field[float]):
            from __gtscript__ import K

            with computation(PARALLEL), interval(K[2], -1):
                field = 0

        module = f"TestIntervalSyntax_error_do_not_mix_{id_version}"
        externals = {}
        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Two-argument syntax"):
            compile_definition(
                definition_func, "test_error_do_not_mix", module, externals=externals
            )

    def test_reversed_interval(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(-1, 1):
                field = 0

        module = f"TestIntervalSyntax_bad_interval_{id_version}"
        externals = {}

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="Invalid interval range specification"
        ):
            compile_definition(definition_func, "test_externals", module, externals=externals)


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

        @gtscript.function
        def identity(field_in):
            return field_in

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

    def test_no_nested_function_call(self, id_version):
        @gtscript.function
        def _lap(dx, phi):
            return (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dx * dx)

        def definition_func(phi: gtscript.Field[np.float64], dx: float):
            from __externals__ import lap

            with computation(PARALLEL), interval(...):
                phi = lap(lap(phi, dx), dx)

        module = f"TestExternalsWithSubroutines_test_no_nested_function_call_{id_version}"
        externals = {
            "lap": _lap,
        }

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="in arguments to function calls"):
            compile_definition(
                definition_func, "test_no_nested_function_calls", module, externals=externals
            )


class TestFunctionReturn:
    def test_no_return(self, id_version):
        @gtscript.function
        def _test_no_return(arg):
            arg = 1

        def definition_func(phi: gtscript.Field[np.float64]):
            from __externals__ import test

            with computation(PARALLEL), interval(...):
                phi = test(phi)

        module = f"TestFunctionReturn_test_no_return_{id_version}"
        externals = {"test": _test_no_return}

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="should have a single return statement"
        ):
            compile_definition(definition_func, "test_no_return", module, externals=externals)

    def test_number_return_args(self, id_version):
        @gtscript.function
        def _test_return_args(arg):
            return 1, 2

        def definition_func(phi: gtscript.Field[np.float64]):
            from __externals__ import test

            with computation(PARALLEL), interval(...):
                phi = test(phi)

        module = f"TestFunctionReturn_test_number_return_args_{id_version}"
        externals = {"test": _test_return_args}

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Number of returns values does not match arguments on left side",
        ):
            compile_definition(
                definition_func, "test_number_return_args", module, externals=externals
            )

    def test_multiple_return(self, id_version):
        @gtscript.function
        def _test_multiple_return(arg):
            return 1
            return 2

        def definition_func(phi: gtscript.Field[np.float64]):
            from __externals__ import test

            with computation(PARALLEL), interval(...):
                phi = test(phi)

        module = f"TestFunctionReturn_test_multiple_return_{id_version}"
        externals = {"test": _test_multiple_return}

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="should have a single return statement"
        ):
            compile_definition(definition_func, "test_multiple_return", module, externals=externals)

    def test_conditional_return(self, id_version):
        @gtscript.function
        def _test_conditional_return(arg):
            if arg > 1:
                tmp = 1
            else:
                tmp = 2
            return tmp

        def definition_func(phi: gtscript.Field[np.float64]):
            from __externals__ import test

            with computation(PARALLEL), interval(...):
                phi = test(phi)

        module = f"TestFunctionReturn_test_conditional_return_{id_version}"
        externals = {"test": _test_conditional_return}

        compile_definition(definition_func, "test_conditional_return", module, externals=externals)


class TestCompileTimeAssertions:
    def test_nomsg(self, id_version):
        def definition(inout_field: gtscript.Field[float]):
            from __externals__ import EXTERNAL

            with computation(PARALLEL), interval(...):
                assert __INLINED(EXTERNAL < 1)
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        module = f"TestCompileTimeAssertions_test_module_{id_version}"
        compile_definition(definition, "test_assert_nomsg", module, externals={"EXTERNAL": 0})

        with pytest.raises(gt_frontend.GTScriptAssertionError, match="Assertion failed"):
            compile_definition(definition, "test_assert_nomsg", module, externals={"EXTERNAL": 1})

    def test_msg(self, id_version):
        def definition(inout_field: gtscript.Field[float]):
            from __externals__ import EXTERNAL

            with computation(PARALLEL), interval(...):
                assert __INLINED(EXTERNAL < 1), "An error occurred"
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        module = f"TestCompileTimeAssertions_test_module_{id_version}"
        with pytest.raises(gt_frontend.GTScriptAssertionError, match="An error occurred"):
            compile_definition(definition, "test_assert_msg", module, externals={"EXTERNAL": 1})

    def test_nested_attribute(self, id_version):
        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                assert __INLINED(GLOBAL_VERY_NESTED_CONSTANTS.nested.A > 1), "An error occurred"
                inout_field = inout_field[0, 0, 0] + GLOBAL_VERY_NESTED_CONSTANTS.nested.A

        module = f"TestCompileTimeAssertions_test_module_{id_version}"
        compile_definition(definition, "test_assert_nested_attribute", module)

    def test_inside_func(self, id_version):
        @gtscript.function
        def assert_in_func(field):
            assert __INLINED(GLOBAL_CONSTANT < 2), "An error occurred"
            return field[0, 0, 0] + GLOBAL_CONSTANT

        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = assert_in_func(inout_field)

        module = f"TestCompileTimeAssertions_test_module_{id_version}"
        compile_definition(definition, "test_inside_func", module)

    def test_runtime_error(self, id_version):
        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                assert __INLINED(inout_field[0, 0, 0] < 0), "An error occurred"

        module = f"TestCompileTimeAssertions_test_module_{id_version}"
        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Evaluation of compile-time assertion condition failed",
        ):
            compile_definition(definition, "test_definition_error", module)


class TestReducedDimensions:
    def test_syntax(self, id_version):
        def definition_func(
            field_3d: gtscript.Field[np.float_, gtscript.IJK],
            field_2d: gtscript.Field[np.float_, gtscript.IJ],
            field_1d: gtscript.Field[np.float_, gtscript.K],
        ):
            with computation(FORWARD), interval(...):
                field_2d = field_1d[1]
                field_3d = field_2d + field_1d

        module = f"TestReducedDimensions_test_syntax_{id_version}"
        externals = {}
        stencil_id, def_ir = compile_definition(
            definition_func, "test_syntax", module, externals=externals
        )

        assert len(def_ir.computations) == 1
        first_stmt = def_ir.computations[0].body.stmts[0]

        value_ref = first_stmt.value
        assert value_ref.name == "field_1d"
        assert set(value_ref.offset.keys()) == {"K"}

        target_ref = first_stmt.target
        assert target_ref.name == "field_2d"
        assert set(target_ref.offset.keys()) == {"I", "J"}

        second_stmt = def_ir.computations[0].body.stmts[1]

        target_ref = second_stmt.target
        assert target_ref.name == "field_3d"
        assert set(target_ref.offset.keys()) == {"I", "J", "K"}

    def test_error_syntax(self, id_version):
        module = f"TestReducedDimensions_test_error_syntax_{id_version}"
        externals = {}

        def definition(
            field_in: gtscript.Field[np.float_, gtscript.K],
            field_out: gtscript.Field[np.float_, gtscript.IJK],
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[0, 0, 1]

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Incorrect offset .* to field .* with dimensions .*",
        ):
            compile_definition(definition, "test_error_syntax", module, externals=externals)

    def test_error_write_1d(self, id_version):
        module = f"TestReducedDimensions_test_error_write_1d_{id_version}"
        externals = {}

        def definition(
            field_in: gtscript.Field[np.float_, gtscript.IJK],
            field_out: gtscript.Field[np.float_, gtscript.K],
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[0, 0, 0]

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Cannot assign to field .* as all parallel axes .* are not present",
        ):
            compile_definition(definition, "test_error_annotation", module, externals=externals)


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

    def test_nested_with_ordering(self):
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
                with interval(0, 1):
                    in_field = out_field + 2
                with interval(1, 2):
                    in_field = out_field + 1

        definitions = {"fw": definition_fw, "bw": definition_bw}

        for name, definition in definitions.items():
            with pytest.raises(
                gt_frontend.GTScriptSyntaxError,
                match=r"(.*?)Intervals must be specified in order of execution(.*)",
            ):
                # generate DIR
                _, definition_ir = compile_definition(
                    definition,
                    f"test_nested_with_reordering_{name}",
                    f"TestImports_test_module_{id_version}",
                )


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
        @gtscript.function
        def sinus(field_in):
            return sin(field_in)

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


class TestAnnotations:
    @staticmethod
    def sumdiff_defs(
        in_a: gtscript.Field["dtype_in"],
        in_b: gtscript.Field["dtype_in"],
        out_c: gtscript.Field["dtype_out"],
        out_d: gtscript.Field[float],
        *,
        wa: "dtype_scalar",
        wb: int,
    ):
        with computation(PARALLEL), interval(...):
            out_c = wa * in_a + wb * in_b
            out_d = wa * in_a - wb * in_b

    @pytest.mark.parametrize("dtype_in", [int, np.float32, np.float64])
    @pytest.mark.parametrize("dtype_out", [int, np.float32, np.float64])
    @pytest.mark.parametrize("dtype_scalar", [int, np.float32, np.float64])
    def test_set_arg_dtypes(self, dtype_in, dtype_out, dtype_scalar):
        definition = self.sumdiff_defs
        dtypes = {"dtype_in": dtype_in, "dtype_out": dtype_out, "dtype_scalar": dtype_scalar}

        definition, original_annotations = gtscript._set_arg_dtypes(definition, dtypes)

        assert "in_a" in original_annotations
        assert isinstance(original_annotations["in_a"], gtscript._FieldDescriptor)
        assert original_annotations["in_a"].dtype == "dtype_in"
        assert "in_b" in original_annotations
        assert isinstance(original_annotations["in_b"], gtscript._FieldDescriptor)
        assert original_annotations["in_b"].dtype == "dtype_in"
        assert "out_c" in original_annotations
        assert isinstance(original_annotations["out_c"], gtscript._FieldDescriptor)
        assert original_annotations["out_c"].dtype == "dtype_out"
        assert "out_d" in original_annotations
        assert isinstance(original_annotations["out_d"], gtscript._FieldDescriptor)
        assert original_annotations["out_d"].dtype == float
        assert "wa" in original_annotations
        assert original_annotations["wa"] == "dtype_scalar"
        assert "wb" in original_annotations
        assert original_annotations["wb"] == int
        assert len(original_annotations) == 6

        annotations = getattr(definition, "__annotations__", {})
        assert "in_a" in annotations
        assert isinstance(annotations["in_a"], gtscript._FieldDescriptor)
        assert annotations["in_a"].dtype == dtype_in
        assert "in_b" in annotations
        assert isinstance(annotations["in_b"], gtscript._FieldDescriptor)
        assert annotations["in_b"].dtype == dtype_in
        assert "out_c" in annotations
        assert isinstance(annotations["out_c"], gtscript._FieldDescriptor)
        assert annotations["out_c"].dtype == dtype_out
        assert "out_d" in annotations
        assert isinstance(annotations["out_d"], gtscript._FieldDescriptor)
        assert annotations["out_d"].dtype == float
        assert "wa" in annotations
        assert annotations["wa"] == dtype_scalar
        assert "wb" in annotations
        assert annotations["wb"] == int
        assert len(annotations) == 6

        setattr(definition, "__annotations__", original_annotations)

    @pytest.mark.parametrize("dtype_in", [int, np.float32, np.float64])
    @pytest.mark.parametrize("dtype_out", [int, np.float32, np.float64])
    @pytest.mark.parametrize("dtype_scalar", [int, np.float32, np.float64])
    def test_compilation(self, dtype_in, dtype_out, dtype_scalar):
        definition = self.sumdiff_defs
        dtypes = {"dtype_in": dtype_in, "dtype_out": dtype_out, "dtype_scalar": dtype_scalar}

        sumdiff = gtscript.stencil("debug", definition, dtypes=dtypes)

        annotations = getattr(definition, "__annotations__", {})
        assert "in_a" in annotations
        assert isinstance(annotations["in_a"], gtscript._FieldDescriptor)
        assert annotations["in_a"].dtype == "dtype_in"
        assert "in_b" in annotations
        assert isinstance(annotations["in_b"], gtscript._FieldDescriptor)
        assert annotations["in_b"].dtype == "dtype_in"
        assert "out_c" in annotations
        assert isinstance(annotations["out_c"], gtscript._FieldDescriptor)
        assert annotations["out_c"].dtype == "dtype_out"
        assert "out_d" in annotations
        assert isinstance(annotations["out_d"], gtscript._FieldDescriptor)
        assert annotations["out_d"].dtype == float
        assert "wa" in annotations
        assert annotations["wa"] == "dtype_scalar"
        assert "wb" in annotations
        assert annotations["wb"] == int
        assert len(annotations) == 6
