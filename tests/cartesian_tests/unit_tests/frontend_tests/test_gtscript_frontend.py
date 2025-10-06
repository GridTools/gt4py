# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import textwrap
import types
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import pytest

import gt4py.cartesian.definitions as gt_definitions
from gt4py.cartesian import gtscript
from gt4py.cartesian.frontend import gtscript_frontend as gt_frontend, nodes
from gt4py.cartesian.frontend.exceptions import GTScriptSyntaxError
from gt4py.cartesian.gtscript import (
    FORWARD,
    IJ,
    IJK,
    PARALLEL,
    Field,
    I,
    J,
    K,
    abs,
    asin,
    compile_assert,
    computation,
    horizontal,
    interval,
    isfinite,
    region,
    sin,
)


# ---- Utilities -----


def parse_definition(
    definition_func: Callable[..., None],
    *,
    name: str,
    module: str,
    externals: Optional[Dict[str, Any]] = None,
    dtypes: Dict[Type, Type] = None,
    literal_int_precision: int | None = None,
    literal_float_precision: int | None = None,
    rebuild=False,
    **kwargs,
) -> nodes.StencilDefinition:
    original_annotations = gtscript._set_arg_dtypes(definition_func, dtypes=dtypes or {})

    build_args = {
        "name": name,
        "module": module,
        "rebuild": rebuild,
        "build_info": None,
        "backend_opts": kwargs,
    }
    if literal_int_precision is not None:
        build_args["literal_int_precision"] = literal_int_precision
    if literal_float_precision is not None:
        build_args["literal_float_precision"] = literal_float_precision

    build_options = gt_definitions.BuildOptions(**build_args)

    gt_frontend.GTScriptFrontend.prepare_stencil_definition(
        definition_func, externals=externals or {}
    )
    definition_ir = gt_frontend.GTScriptParser(
        definition_func, externals=externals or {}, options=build_options, dtypes=dtypes
    ).run("numpy")

    setattr(definition_func, "__annotations__", original_annotations)

    return definition_ir


# ---- Tests -----

GLOBAL_BOOL_CONSTANT = True
GLOBAL_CONSTANT = 1.0
GLOBAL_NESTED_CONSTANTS = types.SimpleNamespace(A=100, B=200)
GLOBAL_VERY_NESTED_CONSTANTS = types.SimpleNamespace(nested=types.SimpleNamespace(A=1000, B=2000))


@gtscript.function
def add_external_const(a):
    return a + 10.0 + GLOBAL_CONSTANT


class TestInlinedExternals:
    def test_all_legal_combinations(self):
        def definition_func(inout_field: gtscript.Field[float]):
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

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_missing(self):
        def definition_func(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptSymbolError, match=r".*MISSING_CONSTANT.*"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

        def definition_func(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + GLOBAL_NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError,
            match=r".*GLOBAL_NESTED_CONSTANTS.missing.*",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_recursive_function_imports(self):
        @gtscript.function
        def func_deeply_nested():
            from gt4py.cartesian.__externals__ import another_const

            return another_const

        @gtscript.function
        def func_nested():
            from gt4py.cartesian.__externals__ import const

            return const + func_deeply_nested()

        @gtscript.function
        def func():
            from gt4py.cartesian.__externals__ import other_call

            return other_call()

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import some_call

            with computation(PARALLEL), interval(...):
                inout_field = func() + some_call()

        def_ir = parse_definition(
            definition_func,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
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
            "test_gtscript_frontend.func_nested.func_deeply_nested",
        }

    def test_decorated_freeze(self):
        A = 0

        @gtscript.function
        def some_function():
            return A

        A = 1

        def definition_func(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = some_function()

        def_ir = parse_definition(
            definition_func,
            externals={"func": some_function},
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

        stmt = def_ir.computations[0].body.stmts[0]
        assert isinstance(stmt.value, nodes.ScalarLiteral) and stmt.value.value == 0

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, value_type):
        WRONG_VALUE_CONSTANT = value_type()

        def definition_func(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_np_bool_external(self):
        def stencil(input_field: gtscript.Field[float], output_field: gtscript.Field[float]):
            from __externals__ import flag

            with computation(PARALLEL), interval(...):
                add_me = 1 if flag else 5
                output_field = input_field + add_me

        def_ir = parse_definition(
            stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            externals={
                "flag": np.bool_(True),
            },
        )

        assert set(def_ir.externals.keys()) == {"flag"}


class TestFunction:
    def test_error_invalid(self):
        def func():
            return 1.0

        def definition_func(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = func()

        with pytest.raises(TypeError, match=r"func is not a gtscript function"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_use_in_expr(self):
        @gtscript.function
        def func():
            return 1.0

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func() + 1

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_use_as_func_arg(self):
        @gtscript.function
        def func():
            return 1.0

        @gtscript.function
        def func_outer(arg):
            return arg + 1

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func_outer(func())

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_use_in_expr_in_func_arg(self):
        @gtscript.function
        def func():
            return 1.0

        @gtscript.function
        def func_outer(arg):
            return arg + 1

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func_outer(func() + 1)

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_use_in_expr_multiple_return(self):
        @gtscript.function
        def func():
            tmp1 = 1
            tmp2 = 2
            return tmp1, tmp2

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func() + 1

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Only functions with a single return value can be used in expressions, including as call arguments. "
            "Please assign the function results to symbols first.",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_use_in_call_arg_multiple_return(self):
        @gtscript.function
        def func():
            tmp1 = 1
            tmp2 = 2
            return tmp1, tmp2

        @gtscript.function
        def func_outer(arg):
            return arg + 1

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func_outer(func())

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Only functions with a single return value can be used in expressions, including as call arguments. "
            "Please assign the function results to symbols first.",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_recursive_function_call_two_externals(self):
        @gtscript.function
        def func1(arg):
            from __externals__ import func2

            return func2(arg)

        @gtscript.function
        def func2(arg):
            from __externals__ import func1

            return func1(arg)

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__gtscript__ import PARALLEL, computation, interval

            with computation(PARALLEL), interval(...):
                inout_field = func2(inout_field)

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="recursive function call"):
            parse_definition(
                definition_func,
                name="testf",
                module=__name__,
                externals={"func1": func1, "func2": func2},
            )

    def test_recursive_function_calls_external_self(self):
        @gtscript.function
        def recursive_fcn(arg):
            from gt4py.cartesian.__externals__ import func

            return func(arg + 1)

        def definition_func(phi: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                phi = recursive_fcn(phi)

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="recursive imports"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
                externals={"func": recursive_fcn},
            )


class TestAxisSyntax:
    def test_good_syntax(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field[J - 1] + in_field[J]

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_good_syntax_external(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import AXIS

            with computation(PARALLEL), interval(...):
                out_field = in_field[AXIS - 1]

        parse_definition(
            definition_func,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            externals={"AXIS": gtscript.Axis("I")},
        )

    def test_good_syntax_external_value(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import VALUE

            with computation(PARALLEL), interval(...):
                out_field = in_field[J - VALUE]

        for value in range(2):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
                externals={"VALUE": value},
            )

    def test_bad_mul_syntax(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field[I * 1]

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_bad_dup_add(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field[I + 1 + I]

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_bad_dup_axis(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field[I, I - 1]

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_bad_out_of_order(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field[J, I - 1]

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )


class TestImportedExternals:
    def test_all_legal_combinations(self):
        externals = dict(
            BOOL_CONSTANT=-1.0,
            CONSTANT=-2.0,
            NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200),
            VERY_NESTED_CONSTANTS=types.SimpleNamespace(
                nested=types.SimpleNamespace(A=-1000, B=-2000)
            ),
        )

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import (
                BOOL_CONSTANT,
                CONSTANT,
                NESTED_CONSTANTS,
                VERY_NESTED_CONSTANTS,
            )

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

        parse_definition(
            definition_func,
            externals=externals,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

    def test_missing(self):
        externals = dict(CONSTANT=-2.0, NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200))

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import MISSING_CONSTANT

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*MISSING_CONSTANT.*"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import NESTED_CONSTANTS

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError, match=r".*NESTED_CONSTANTS.missing.*"
        ):
            parse_definition(
                definition_func,
                externals=externals,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, value_type):
        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import WRONG_VALUE_CONSTANT

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        externals = dict(WRONG_VALUE_CONSTANT=value_type())

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            parse_definition(
                definition_func,
                externals=externals,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )


class TestIntervalSyntax:
    def test_simple(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(0, 1):
                field = 0

        def_ir = parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == nodes.AxisBound(
            level=nodes.LevelMarker.START, offset=0, loc=loc
        )
        assert def_ir.computations[0].interval.end == nodes.AxisBound(
            level=nodes.LevelMarker.START, offset=1, loc=loc
        )

    def test_none(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(1, None):
                field = 0

        def_ir = parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == nodes.AxisBound(
            level=nodes.LevelMarker.START, offset=1, loc=loc
        )
        assert def_ir.computations[0].interval.end == nodes.AxisBound(
            level=nodes.LevelMarker.END, offset=0, loc=loc
        )

    def test_externals(self):
        def definition_func(field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import kstart

            with computation(PARALLEL), interval(kstart, -1):
                field = 0

        for kstart in (3, gtscript.K[3]):
            # An implementation quirk allows us to use gtscript.K[3] here,
            # although it is not great form to do so, since two-argument syntax
            # should not use AxisIndex objects.
            def_ir = parse_definition(
                definition_func,
                externals={"kstart": kstart},
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )
            loc = def_ir.computations[0].interval.loc
            assert def_ir.computations[0].interval.start == nodes.AxisBound(
                level=nodes.LevelMarker.START, offset=3, loc=loc
            )
            assert def_ir.computations[0].interval.end == nodes.AxisBound(
                level=nodes.LevelMarker.END, offset=-1, loc=loc
            )

    def test_axisinterval(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(K[1:-1]):
                field = 0

        def_ir = parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )
        loc = def_ir.computations[0].interval.loc
        assert def_ir.computations[0].interval.start == nodes.AxisBound(
            level=nodes.LevelMarker.START, offset=1, loc=loc
        )
        assert def_ir.computations[0].interval.end == nodes.AxisBound(
            level=nodes.LevelMarker.END, offset=-1, loc=loc
        )

    def test_error_none(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(None, -1):
                field = 0

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Invalid interval range specification",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_error_do_not_mix(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(K[2], -1):
                field = 0

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Two-argument syntax"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_reversed_interval(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL), interval(-1, 1):
                field = 0

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Invalid interval range specification",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_overlapping_intervals_none(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL):
                with interval(0, None):
                    field = 0
                with interval(-1, None):
                    field = 1

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Overlapping intervals"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_overlapping_intervals(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL):
                with interval(0, 3):
                    field = 0
                with interval(2, None):
                    field = 1

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Overlapping intervals"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_nonoverlapping_intervals(self):
        def definition_func(field: gtscript.Field[float]):
            with computation(PARALLEL):
                with interval(0, 2):
                    field = 0
                with interval(3, -1):
                    field = 1
                with interval(-1, None):
                    field = 2

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )


class TestRegions:
    def test_one_interval_only(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...), horizontal(region[I[0:3], :]):
                in_f = 1.0

        def_ir = parse_definition(
            stencil, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

        assert len(def_ir.computations) == 1
        assert isinstance(def_ir.computations[0].body.stmts[0], nodes.HorizontalIf)

    def test_one_interval_only_single(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...), horizontal(region[I[0], :]):
                in_f = 1.0

        def_ir = parse_definition(
            stencil, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

        assert len(def_ir.computations) == 1
        assert def_ir.computations[0].body.stmts[0].intervals["I"].is_single_index

    def test_from_external(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            from gt4py.cartesian.__externals__ import i1

            with computation(PARALLEL), interval(...), horizontal(region[i1, :]):
                in_f = 1.0

        def_ir = parse_definition(
            stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            externals={"i1": I[1]},
        )

        assert len(def_ir.computations) == 1
        assert (
            def_ir.computations[0].body.stmts[0].intervals["I"].start.level
            == nodes.LevelMarker.START
        )
        assert def_ir.computations[0].body.stmts[0].intervals["I"].start.offset == 1
        assert def_ir.computations[0].body.stmts[0].intervals["I"].is_single_index

    def test_multiple_inline(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with horizontal(region[I[0], :], region[:, J[-1]]):
                    in_f = 1.0

        def_ir = parse_definition(
            stencil, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

        assert len(def_ir.computations[0].body.stmts) == 3

    def test_inside_function(self):
        @gtscript.function
        def region_func():
            from gt4py.cartesian.__externals__ import ie

            field = 0.0
            with horizontal(region[ie, :]):
                field = 1.0

            return field

        def stencil(in_f: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_f = region_func()

        def_ir = parse_definition(
            stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            externals={"ie": I[-1]},
        )

    def test_error_undefined(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            from gt4py.cartesian.__externals__ import i0  # forget to add 'ia'

            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with horizontal(region[i0 : 1 + ia, :]):
                    in_f = 1.0

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Unknown symbol"):
            parse_definition(stencil, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_error_nested(self):
        def stencil(in_f: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_f = in_f + 1.0
                with horizontal(region[I[0], :]):
                    in_f = 1.0
                    with horizontal(region[:, J[-1]]):
                        in_f = 2.0

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Cannot nest `with` node inside a horizontal region.",
        ):
            parse_definition(stencil, name=inspect.stack()[0][3], module=self.__class__.__name__)


class TestExternalsWithSubroutines:
    def test_all_legal_combinations(self):
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
            from gt4py.cartesian.__externals__ import (
                stage_laplacian_x,
                stage_laplacian_y,
            )

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
            from gt4py.cartesian.__externals__ import (
                stage_laplacian,
                stage_laplacian_x,
                stage_laplacian_y,
            )
            from gt4py.cartesian.__gtscript__ import (
                BACKWARD,
                FORWARD,
                PARALLEL,
                computation,
                interval,
            )

            with computation(PARALLEL), interval(...):
                lap = stage_laplacian(dx=dx, dy=dy, phi=in_phi) + GLOBAL_CONSTANT
                out_phi = in_gamma[0, 0, 0] * lap[0, 0, 0]

            with computation(PARALLEL), interval(...):
                tmp_out = identity(in_phi)
                out_phi = tmp_out + 1

            with computation(PARALLEL), interval(...):
                tmp_out2 = identity(in_gamma)
                out_field = out_phi + tmp_out2

        externals = {
            "stage_laplacian": _stage_laplacian,
            "stage_laplacian_x": _stage_laplacian_x,
            "stage_laplacian_y": _stage_laplacian_y,
        }
        parse_definition(
            definition_func,
            externals=externals,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )


class TestFunctionReturn:
    def test_no_return(self):
        @gtscript.function
        def test_no_return(arg):
            arg = 1

        def definition_func(phi: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                phi = test_no_return(phi)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="should have a single return statement",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_number_return_args(self):
        @gtscript.function
        def test_return_args(arg):
            return 1, 2

        def definition_func(phi: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                phi = test_return_args(phi)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Number of returns values does not match arguments on left side",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_multiple_return(self):
        @gtscript.function
        def test_multiple_return(arg):
            return 1
            return 2

        def definition_func(phi: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                phi = test_multiple_return(phi)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="should have a single return statement",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_conditional_return(self):
        @gtscript.function
        def test_conditional_return(arg):
            if arg > 1:
                tmp = 1
            else:
                tmp = 2
            return tmp

        def definition_func(phi: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                phi = test_conditional_return(phi)

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_return_tuple(self):
        @gtscript.function
        def return_tuple():
            tmp1 = 1
            tmp2 = 2
            return tmp1, tmp2

        def definition_func(res1: gtscript.Field[np.float64], res2: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                res1, res2 = return_tuple()

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )


class TestCompileTimeAssertions:
    def test_nomsg(self):
        def definition(inout_field: gtscript.Field[float]):
            from gt4py.cartesian.__externals__ import EXTERNAL

            with computation(PARALLEL), interval(...):
                compile_assert(EXTERNAL < 1)
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        parse_definition(
            definition,
            externals={"EXTERNAL": 0},
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

        with pytest.raises(gt_frontend.GTScriptAssertionError, match="Assertion failed"):
            parse_definition(
                definition,
                externals={"EXTERNAL": 1},
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_nested_attribute(self):
        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                compile_assert(GLOBAL_VERY_NESTED_CONSTANTS.nested.A > 1)
                inout_field = inout_field[0, 0, 0] + GLOBAL_VERY_NESTED_CONSTANTS.nested.A

        parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_inside_func(self):
        @gtscript.function
        def assert_in_func(field):
            compile_assert(GLOBAL_CONSTANT < 2)
            return field[0, 0, 0] + GLOBAL_CONSTANT

        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                inout_field = assert_in_func(inout_field)

        parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_runtime_error(self):
        def definition(inout_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                compile_assert(inout_field[0, 0, 0] < 0)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match="Evaluation of compile_assert condition failed"
        ):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)


class TestReducedDimensions:
    def test_syntax(self):
        def definition_func(
            field_3d: gtscript.Field[gtscript.IJK, np.float64],
            field_2d: gtscript.Field[gtscript.IJ, np.float64],
            field_1d: gtscript.Field[gtscript.K, np.float64],
        ):
            with computation(FORWARD), interval(...):
                field_2d = field_1d[1]
                field_3d = field_2d + field_1d

        def_ir = parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
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

    def test_error_syntax(self):
        def definition(
            field_in: gtscript.Field[gtscript.K, np.float64],
            field_out: gtscript.Field[gtscript.IJK, np.float64],
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[0, 0, 1]

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Incorrect offset specification detected for .*. Found .* but .* has dimensions .*",
        ):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_error_write_1d(self):
        def definition(
            field_in: gtscript.Field[gtscript.IJK, np.float64],
            field_out: gtscript.Field[gtscript.K, np.float64],
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[0, 0, 0]

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Cannot assign to field .* as all parallel axes .* are not present",
        ):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_higher_dim_temp(self):
        def definition(
            field_in: gtscript.Field[gtscript.IJK, np.float64],
            field_out: gtscript.Field[gtscript.IJK, np.float64],
        ):
            tmp: Field[IJK, (np.float64, (2,))] = 0.0
            with computation(PARALLEL), interval(...):
                tmp[0, 0, 0][0] = field_in
                field_out = tmp[0, 0, 0][0]

        parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_typed_temp_missing(self):
        def definition(
            field_in: gtscript.Field[gtscript.IJK, np.float64],
            field_out: gtscript.Field[gtscript.IJK, np.float64],
        ):
            tmp: Field[IJ, np.float64] = 0.0
            with computation(FORWARD), interval(1, None):
                tmp = field_in[0, 0, -1]
                field_out = tmp

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Found IJ, but only IJK is currently supported for temporaries",
        ):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)


class TestDataDimensions:
    def test_syntax(self):
        def definition(
            field_in: gtscript.Field[np.float64],
            another_field: gtscript.Field[(np.float64, 3)],
            field_out: gtscript.Field[gtscript.IJK, (np.float64, (3,))],
        ):
            with computation(PARALLEL), interval(...):
                field_out[0, 0, 0][0] = field_in
                field_out[0, 0, 0][1] = field_in
                field_out[0, 0, 0][2] = field_in[0, 0, 0] + another_field[0, 0, 0][2]

        parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_syntax_no_datadim(self):
        def definition(
            field_in: gtscript.Field[np.float64],
            field_out: gtscript.Field[gtscript.IJK, (np.float64, (3,))],
        ):
            with computation(PARALLEL), interval(...):
                field_out[0, 0, 0][0] = field_in
                field_out[0, 0, 0][1] = field_in
                field_out[0, 0, 0][2] = field_in[0, 0, 0][0]

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Incorrect data index length"):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_syntax_out_bounds(self):
        def definition(
            field_in: gtscript.Field[np.float64],
            field_out: gtscript.Field[gtscript.IJK, (np.float64, (3,))],
        ):
            with computation(PARALLEL), interval(...):
                field_out[0, 0, 0][3] = field_in[0, 0, 0]

        with pytest.raises(gt_frontend.GTScriptSyntaxError, match="Data index out of bounds"):
            parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_indirect_access_read(self):
        def definition(
            field_3d: gtscript.Field[np.float64],
            field_4d: gtscript.Field[gtscript.IJK, (np.float64, (2,))],
            variable: float,
        ):
            with computation(PARALLEL), interval(...):
                field_3d = field_4d[0, 0, 0][variable]

        def_ir = parse_definition(
            definition, name=inspect.stack()[0][3], module=self.__class__.__name__
        )
        assert isinstance(def_ir.computations[0].body.stmts[0].value.data_index[0], nodes.VarRef)

    def test_indirect_access_write(self):
        def definition(
            field_3d: gtscript.Field[np.float64],
            field_4d: gtscript.Field[gtscript.IJK, (np.float64, (2,))],
            variable: float,
        ):
            with computation(PARALLEL), interval(...):
                field_4d[0, 0, 0][variable] = field_3d

        def_ir = parse_definition(
            definition, name=inspect.stack()[0][3], module=self.__class__.__name__
        )
        assert isinstance(def_ir.computations[0].body.stmts[0].target.data_index[0], nodes.VarRef)


class TestImports:
    def test_all_legal_combinations(self):
        def definition_func(inout_field: gtscript.Field[float]):
            from __externals__ import EXTERNAL
            from __gtscript__ import BACKWARD, FORWARD, PARALLEL, computation, interval
            from gt4py.cartesian.__externals__ import EXTERNAL
            from gt4py.cartesian.__gtscript__ import (
                BACKWARD,
                FORWARD,
                PARALLEL,
                computation,
                interval,
            )

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        parse_definition(
            definition_func,
            externals={"EXTERNAL": 1.0},
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

    @pytest.mark.parametrize(
        "id_case,import_line",
        list(
            enumerate(
                [
                    "import gt4py",
                    "from externals import EXTERNAL",
                    "from gt4py.cartesian import __gtscript__",
                    "from gt4py.cartesian import __externals__",
                    "from gt4py.cartesian.gtscript import computation",
                    "from gt4py.cartesian.externals import EXTERNAL",
                ]
            )
        ),
    )
    def test_wrong_imports(self, id_case, import_line):
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
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )


class TestDTypes:
    @pytest.mark.parametrize(
        "id_case,test_dtype",
        list(
            enumerate(
                [
                    bool,
                    np.bool_,
                    int,
                    np.int32,
                    np.int64,
                    float,
                    np.float32,
                    np.float64,
                    np.dtype((np.float32, (3,))),
                ]
            )
        ),
    )
    def test_all_legal_dtypes_instance(self, id_case, test_dtype):
        test_base_dtype = test_dtype.base.type if isinstance(test_dtype, np.dtype) else test_dtype

        def definition_func(
            in_field: gtscript.Field[test_dtype],
            out_field: gtscript.Field[test_dtype],
            param: test_base_dtype,
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

        def definition_func(
            in_field: gtscript.Field["dtype"],
            out_field: gtscript.Field["dtype"],
            param: "test_base_dtype",
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        parse_definition(
            definition_func,
            dtypes={"dtype": test_dtype, "test_base_dtype": test_base_dtype},
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

    @pytest.mark.parametrize(
        "id_case,test_dtype",
        list(enumerate([str, np.uint32, np.uint64, dict, map, bytes])),
    )
    def test_invalid_inlined_dtypes(self, id_case, test_dtype):
        with pytest.raises(ValueError, match=r".*data type descriptor.*"):

            def definition_func(
                in_field: gtscript.Field[test_dtype],
                out_field: gtscript.Field[test_dtype],
                param: test_dtype,
            ):
                with computation(PARALLEL), interval(...):
                    out_field = in_field + param

    @pytest.mark.parametrize(
        "id_case,test_dtype",
        list(enumerate([str, np.uint32, np.uint64, dict, map, bytes])),
    )
    def test_invalid_external_dtypes(self, id_case, test_dtype):
        def definition_func(
            in_field: gtscript.Field["dtype"],
            out_field: gtscript.Field["dtype"],
            param: "dtype",
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        with pytest.raises(ValueError, match=r".*data type descriptor.*"):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
                dtypes={"dtype": test_dtype},
            )


class TestBuiltinDTypes:
    @staticmethod
    def literal_add_func(in_field: gtscript.Field[float], out_field: gtscript.Field["my_float"]):
        with computation(PARALLEL), interval(...):
            out_field = in_field + 42.0

    @pytest.mark.parametrize("the_float", [np.float32, np.float64])
    def test_literal_floating_parametrization(self, the_float):
        def_ir = parse_definition(
            definition_func=self.literal_add_func,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            dtypes={float: the_float, "my_float": the_float},
        )
        # Check fields dtype are replaced only when str is used
        def_ir.api_fields[0].data_type == nodes.DataType.FLOAT64
        def_ir.api_fields[1].data_type == nodes.DataType.from_dtype(the_float)
        # Check scalar literal dtype is always replaced
        assert def_ir.computations[0].body.stmts[
            0
        ].value.rhs.data_type == nodes.DataType.from_dtype(the_float)


class TestAssignmentSyntax:
    def test_ellipsis(self):
        def func(in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                out_field[...] = in_field

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_offset(self):
        def func(in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                out_field[0, 0, 0] = in_field

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            def func(
                in_field: gtscript.Field[np.float64],
                out_field: gtscript.Field[np.float64],
            ):
                with computation(PARALLEL), interval(...):
                    out_field[0, 0, 1] = in_field

            parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

        def func(in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]):
            from gt4py.cartesian.__externals__ import offset

            with computation(PARALLEL), interval(...):
                out_field[0, 0, offset] = in_field

        parse_definition(
            func,
            externals={"offset": 0},
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Assignment to non-zero offsets in K is not available in PARALLEL. Choose FORWARD or BACKWARD.",
        ):
            parse_definition(
                func,
                externals={"offset": 1},
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_return_to_subscript(self):
        @gtscript.function
        def func(a):
            return a

        def definition_func(
            input_field: gtscript.Field[gtscript.IJK, np.int32],
            output_field: gtscript.Field[gtscript.IJK, np.int32],
        ):
            with computation(PARALLEL), interval(...):
                output_field[0, 0, 0] = func(input_field)

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_return_to_nonzero_subscript(self):
        @gtscript.function
        def func(a):
            return a

        def definition_func(
            input_field: gtscript.Field[gtscript.IJK, np.int32],
            output_field: gtscript.Field[gtscript.IJK, np.int32],
        ):
            with computation(PARALLEL), interval(...):
                output_field[0, 0, 1] = func(input_field)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Assignment to non-zero offsets in K is not available in PARALLEL. Choose FORWARD or BACKWARD.",
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_slice(self):
        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            def func(
                in_field: gtscript.Field[np.float64],
                out_field: gtscript.Field[np.float64],
            ):
                with computation(PARALLEL), interval(...):
                    out_field[:, :, :] = in_field

            parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_string(self):
        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            def func(
                in_field: gtscript.Field[np.float64],
                out_field: gtscript.Field[np.float64],
            ):
                with computation(PARALLEL), interval(...):
                    out_field["a_key"] = in_field

            parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_augmented(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += 2.0
                in_field -= 0.5
                in_field /= 0.5
                in_field *= 4.0

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_K_offset_write(self):
        def func(out: gtscript.Field[np.float64], inp: gtscript.Field[np.float64]):
            with computation(FORWARD), interval(...):
                out[0, 0, 1] = inp

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match=r"(.*?)Assignment to non-zero offsets in K is not available in PARALLEL. Choose FORWARD or BACKWARD.(.*)",
        ):

            def func(out: gtscript.Field[np.float64], inp: gtscript.Field[np.float64]):
                with computation(PARALLEL), interval(...):
                    out[0, 0, 1] = inp

            parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_datadims_direct_access(self):
        # Check classic data dimensions are working
        def data_dims(
            out_field: gtscript.Field[gtscript.IJK, np.int32],
            global_field: gtscript.Field[(np.int32, (3, 3, 3))],
        ):
            with computation(PARALLEL), interval(...):
                out_field = global_field[0, 0, 0][1, 0, 2]

        parse_definition(data_dims, name=inspect.stack()[0][3], module=self.__class__.__name__)

        # Check .A on read
        def at_read(
            out_field: gtscript.Field[gtscript.IJK, np.int32],
            global_field: gtscript.GlobalTable[(np.int32, (3, 3, 3, 3))],
        ):
            with computation(PARALLEL), interval(...):
                out_field = global_field.A[1, 0, 2, 2]

        parse_definition(at_read, name=inspect.stack()[0][3], module=self.__class__.__name__)

        # Can't write to the field
        def at_write(
            in_field: gtscript.Field[gtscript.IJK, np.int32],
            global_field: gtscript.GlobalTable[(np.int32, (3, 3, 3))],
        ):
            with computation(PARALLEL), interval(...):
                global_field.A[1, 0, 2] = in_field

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="writing to an GlobalTable \\('A' global indexation\\) is forbidden",
        ):
            parse_definition(at_write, name=inspect.stack()[0][3], module=self.__class__.__name__)

        # Can't index cartesian style
        def GlobalTable_access_as_IJK(
            out_field: gtscript.Field[gtscript.IJK, np.int32],
            global_field: gtscript.GlobalTable[(np.int32, (3, 3, 3))],
        ):
            with computation(PARALLEL), interval(...):
                out_field = global_field[1, 0, 2]

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="Incorrect offset specification detected for .*. Found .* but .* has dimensions .* Did you mean absolute indexing via .A.*",
        ):
            parse_definition(
                GlobalTable_access_as_IJK,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

        # Check .A on read with a Field with data dimensions
        def data_dims_with_at(
            out_field: gtscript.Field[gtscript.IJK, np.int32],
            global_field: gtscript.Field[(np.int32, (3, 3, 3))],
        ):
            with computation(PARALLEL), interval(...):
                out_field = global_field.A[1, 0, 2]

        parse_definition(
            data_dims_with_at,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )


class TestNestedWithSyntax:
    def test_nested_with(self):
        def definition(in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]):
            with computation(PARALLEL):
                with interval(...):
                    in_field = out_field

        parse_definition(definition, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_nested_with_ordering(self):
        def definition_fw(
            in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
        ):
            from gt4py.cartesian.__gtscript__ import FORWARD, computation, interval

            with computation(FORWARD):
                with interval(1, 2):
                    in_field = out_field + 1
                with interval(0, 1):
                    in_field = out_field + 2

        def definition_bw(
            in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
        ):
            from gt4py.cartesian.__gtscript__ import FORWARD, computation, interval

            with computation(BACKWARD):
                with interval(0, 1):
                    in_field = out_field + 2
                with interval(1, 2):
                    in_field = out_field + 1

        for definition in (definition_fw, definition_bw):
            with pytest.raises(
                gt_frontend.GTScriptSyntaxError,
                match=r"(.*?)Intervals must be specified in order of execution(.*)",
            ):
                parse_definition(
                    definition,
                    name=inspect.stack()[0][3],
                    module=self.__class__.__name__,
                )


class TestNativeFunctions:
    def test_simple_call(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += sin(in_field)

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_offset_arg(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += sin(in_field[1, 0, 0])

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_nested_calls(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += sin(abs(in_field))

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_nested_external_call(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += sin(add_external_const(in_field))

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_multi_nested_calls(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += min(abs(sin(add_external_const(in_field))), -0.5)

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_native_in_function(self):
        @gtscript.function
        def sinus(field_in):
            return sin(field_in)

        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field += sinus(in_field)

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_native_function_unary(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field = not isfinite(in_field)

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_native_function_binary(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field = asin(in_field) + 1

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)

    def test_native_function_ternary(self):
        def func(in_field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                in_field = asin(in_field) + 1 if 1 < in_field else sin(in_field)

        parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)


@gtscript.function
def boolean_return(a):
    return a == 1


class TestFunctionIfError:
    def test_function_if_error(self):
        def func(field: gtscript.Field[np.float64]):  # type: ignore
            with computation(PARALLEL), interval(...):
                field = 0
                if boolean_return(field):
                    field = 1

        with pytest.raises(
            GTScriptSyntaxError,
            match="Using function calls in the condition of an if is not allowed",
        ):
            parse_definition(func, name=inspect.stack()[0][3], module=self.__class__.__name__)


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
        dtypes = {
            "dtype_in": dtype_in,
            "dtype_out": dtype_out,
            "dtype_scalar": dtype_scalar,
        }

        original_annotations = gtscript._set_arg_dtypes(definition, dtypes)

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
    def test_parsing(self, dtype_in, dtype_out, dtype_scalar):
        definition = self.sumdiff_defs
        dtypes = {
            "dtype_in": dtype_in,
            "dtype_out": dtype_out,
            "dtype_scalar": dtype_scalar,
        }

        parse_definition(
            definition,
            dtypes=dtypes,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

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


class TestAbsoluteIndex:
    def test_good_syntax(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field.at(K=0) + in_field.at(K=1)

        parse_definition(
            definition_func, name=inspect.stack()[0][3], module=self.__class__.__name__
        )

    def test_bad_syntax_not_specifying_K(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field.at(2)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match=r".*Absolute K index: Bad syntax.*"
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )

    def test_bad_syntax_specifying_I_J_axis(self):
        def definition_func(in_field: gtscript.Field[float], out_field: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                out_field = in_field.at(I=1, K=0)

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError, match=r".*Absolute K index: Bad syntax.*"
        ):
            parse_definition(
                definition_func,
                name=inspect.stack()[0][3],
                module=self.__class__.__name__,
            )


class TestLiteralCasts:
    def setup_method(self):
        def float_cast(field: gtscript.Field[float]):  # type: ignore
            with computation(PARALLEL), interval(0, 1):
                field[0, 0, 0] = float(0)

        def int_cast(field: gtscript.Field[float]):  # type: ignore
            with computation(PARALLEL), interval(0, 1):
                field[0, 0, 0] = int(0)

        self.float_cast = float_cast
        self.int_cast = int_cast

    def test_explicit_64_int_cast(self):
        def_ir = parse_definition(
            self.int_cast,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_int_precision=64,
        )
        cast_call: nodes.NativeFuncCall = def_ir.computations[0].body.stmts[0].value
        assert cast_call.func == nodes.NativeFunction.INT64

    def test_explicit_32_int_cast(self):
        def_ir = parse_definition(
            self.int_cast,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_int_precision=32,
        )
        cast_call: nodes.NativeFuncCall = def_ir.computations[0].body.stmts[0].value
        assert cast_call.func == nodes.NativeFunction.INT32

    def test_explicit_32_float_cast(self):
        def_ir = parse_definition(
            self.float_cast,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_float_precision=32,
        )
        cast_call: nodes.NativeFuncCall = def_ir.computations[0].body.stmts[0].value
        assert cast_call.func == nodes.NativeFunction.FLOAT32

    def test_explicit_64_float_cast(self):
        def_ir = parse_definition(
            self.float_cast,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_float_precision=64,
        )
        cast_call: nodes.NativeFuncCall = def_ir.computations[0].body.stmts[0].value
        assert cast_call.func == nodes.NativeFunction.FLOAT64


class TestTemporaryTypes:
    def setup_method(self):
        def temporary_int_stencil(field: gtscript.Field[float]):  # type: ignore
            with computation(PARALLEL), interval(0, 1):
                temporary: int = 12
                field[0, 0, 0] = temporary

        def temporary_float_stencil(field: gtscript.Field[float]):  # type: ignore
            with computation(PARALLEL), interval(0, 1):
                temporary: float = 12
                field[0, 0, 0] = temporary

        self.int_stencil = temporary_int_stencil
        self.float_stencil = temporary_float_stencil

    def test_explicit_64_int_temp_defn(self):
        def_ir = parse_definition(
            self.int_stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_int_precision=64,
        )

        field_declaration: nodes.FieldDecl = def_ir.computations[0].body.stmts[0]
        assert field_declaration.data_type == nodes.DataType.INT64

    def test_explicit_32_int_temp_defn(self):
        def_ir = parse_definition(
            self.int_stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_int_precision=32,
        )

        field_declaration: nodes.FieldDecl = def_ir.computations[0].body.stmts[0]
        assert field_declaration.data_type == nodes.DataType.INT32

    def test_explicit_64_float_temp_defn(self):
        def_ir = parse_definition(
            self.float_stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_float_precision=64,
        )

        field_declaration: nodes.FieldDecl = def_ir.computations[0].body.stmts[0]
        assert field_declaration.data_type == nodes.DataType.FLOAT64

    def test_explicit_32_float_temp_defn(self):
        def_ir = parse_definition(
            self.float_stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
            literal_float_precision=32,
        )

        field_declaration: nodes.FieldDecl = def_ir.computations[0].body.stmts[0]
        assert field_declaration.data_type == nodes.DataType.FLOAT32


class TestNumpyTypedConstants:
    def setup_method(self):
        self.constant = np.float32(42.0)

        def assign_constant(field: gtscript.Field[float]):  # type: ignore
            with computation(PARALLEL), interval(0, 1):
                field[0, 0, 0] = self.constant

        self.stencil = assign_constant

    def test_assign_constant_numpy_typed(self):
        def_ir = parse_definition(
            self.stencil,
            name=inspect.stack()[0][3],
            module=self.__class__.__name__,
        )

        constant: nodes.ScalarLiteral = def_ir.computations[0].body.stmts[0].value
        assert isinstance(constant, nodes.ScalarLiteral)
        assert constant.data_type == nodes.DataType.FLOAT32
