# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pytest

from gt4py.next import (
    Dimension,
    DimensionKind,
)
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.iterator.type_system import type_specifications as ts_it

TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


def type_info_cases() -> list[tuple[Optional[ts.TypeSpec], dict]]:
    return [
        (ts.DeferredType(constraint=None), {"is_concrete": False}),
        (
            ts.DeferredType(constraint=ts.ScalarType),
            {"is_concrete": False, "type_class": ts.ScalarType},
        ),
        (
            ts.DeferredType(constraint=ts.FieldType),
            {"is_concrete": False, "type_class": ts.FieldType},
        ),
        (
            ts.ScalarType(kind=ts.ScalarKind.INT64),
            {
                "is_concrete": True,
                "type_class": ts.ScalarType,
                "is_arithmetic": True,
                "is_logical": False,
            },
        ),
        (
            ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)),
            {
                "is_concrete": True,
                "type_class": ts.FieldType,
                "is_arithmetic": False,
                "is_logical": True,
            },
        ),
    ]


def callable_type_info_cases():
    # reuse all the other test cases
    not_callable = [
        (symbol_type, [], {}, [r"Expected a callable type, got "], None)
        for symbol_type, attributes in type_info_cases()
        if not isinstance(symbol_type, ts.CallableType)
    ]

    IDim = Dimension("I")
    JDim = Dimension("J")
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)

    bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    field_type = ts.FieldType(dims=[Dimension("I")], dtype=float_type)
    tuple_type = ts.TupleType(types=[bool_type, field_type])
    nullary_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    unary_func_type = ts.FunctionType(
        pos_only_args=[bool_type], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    kw_only_arg_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={}, kw_only_args={"foo": bool_type}, returns=ts.VoidType()
    )
    kw_or_pos_arg_func_type = ts.FunctionType(
        pos_only_args=[], pos_or_kw_args={"foo": bool_type}, kw_only_args={}, returns=ts.VoidType()
    )
    pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type = ts.FunctionType(
        pos_only_args=[bool_type],
        pos_or_kw_args={"foo": int_type},
        kw_only_args={"bar": float_type},
        returns=ts.VoidType(),
    )
    unary_tuple_arg_func_type = ts.FunctionType(
        pos_only_args=[tuple_type], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )
    fieldop_type = ts_ffront.FieldOperatorType(
        definition=ts.FunctionType(
            pos_only_args=[field_type, float_type],
            pos_or_kw_args={},
            kw_only_args={},
            returns=field_type,
        )
    )
    scanop_type = ts_ffront.ScanOperatorType(
        axis=KDim,
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"carry": float_type, "a": int_type, "b": int_type},
            kw_only_args={},
            returns=float_type,
        ),
    )
    tuple_scanop_type = ts_ffront.ScanOperatorType(
        axis=KDim,
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"carry": float_type, "a": ts.TupleType(types=[int_type, int_type])},
            kw_only_args={},
            returns=float_type,
        ),
    )

    return [
        # func_type, pos_only_args, kwargs, expected incompatibilities, return type
        *not_callable,
        (nullary_func_type, [], {}, [], ts.VoidType()),
        (
            nullary_func_type,
            [bool_type],
            {},
            [r"Function takes 0 positional arguments, but 1 were given."],
            None,
        ),
        (
            nullary_func_type,
            [],
            {"foo": bool_type},
            [r"Got unexpected keyword argument 'foo'."],
            None,
        ),
        (
            unary_func_type,
            [],
            {},
            [r"Function takes 1 positional argument, but 0 were given."],
            None,
        ),
        (unary_func_type, [bool_type], {}, [], ts.VoidType()),
        (
            unary_func_type,
            [float_type],
            {},
            [r"Expected 1st argument to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_or_pos_arg_func_type,
            [],
            {},
            [
                r"Missing 1 required positional argument: 'foo'",
                r"Function takes 1 positional argument, but 0 were given.",
            ],
            None,
        ),
        # function with keyword-or-positional argument
        (kw_or_pos_arg_func_type, [], {"foo": bool_type}, [], ts.VoidType()),
        (
            kw_or_pos_arg_func_type,
            [],
            {"foo": float_type},
            [r"Expected argument 'foo' to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_or_pos_arg_func_type,
            [],
            {"bar": bool_type},
            [r"Got unexpected keyword argument 'bar'."],
            None,
        ),
        # function with keyword-only argument
        (kw_only_arg_func_type, [], {}, [r"Missing required keyword argument 'foo'."], None),
        (kw_only_arg_func_type, [], {"foo": bool_type}, [], ts.VoidType()),
        (
            kw_only_arg_func_type,
            [],
            {"foo": float_type},
            [r"Expected keyword argument 'foo' to be of type 'bool', got 'float64'."],
            None,
        ),
        (
            kw_only_arg_func_type,
            [],
            {"bar": bool_type},
            [r"Got unexpected keyword argument 'bar'."],
            None,
        ),
        # function with positional, keyword-or-positional, and keyword-only argument
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [],
            {},
            [
                r"Missing 1 required positional argument: 'foo'",
                r"Function takes 2 positional arguments, but 0 were given.",
                r"Missing required keyword argument 'bar'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {},
            [
                r"Function takes 2 positional arguments, but 1 were given.",
                r"Missing required keyword argument 'bar'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"foo": int_type},
            [r"Missing required keyword argument 'bar'"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"foo": int_type},
            [r"Missing required keyword argument 'bar'"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type, bool_type],
            {"bar": float_type, "foo": int_type},
            [r"G"],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [int_type],
            {"bar": bool_type, "foo": bool_type},
            [
                r"Expected 1st argument to be of type 'bool', got 'int64'",
                r"Expected argument 'foo' to be of type 'int64', got 'bool'",
                r"Expected keyword argument 'bar' to be of type 'float64', got 'bool'",
            ],
            None,
        ),
        (
            pos_arg_and_kw_or_pos_arg_and_kw_only_arg_func_type,
            [bool_type],
            {"bar": float_type, "foo": int_type},
            [],
            ts.VoidType(),
        ),
        (unary_tuple_arg_func_type, [tuple_type], {}, [], ts.VoidType()),
        (
            unary_tuple_arg_func_type,
            [ts.TupleType(types=[float_type, field_type])],
            {},
            [
                r"Expected 1st argument to be of type 'tuple\[bool, Field\[\[I\], float64\]\]', got 'tuple\[float64, Field\[\[I\], float64\]\]'"
            ],
            ts.VoidType(),
        ),
        (
            unary_tuple_arg_func_type,
            [int_type],
            {},
            [
                r"Expected 1st argument to be of type 'tuple\[bool, Field\[\[I\], float64\]\]', got 'int64'"
            ],
            ts.VoidType(),
        ),
        # field operator
        (fieldop_type, [field_type, float_type], {}, [], field_type),
        # scan operator
        (
            scanop_type,
            [],
            {},
            [r"Scan operator takes 2 positional arguments, but 0 were given."],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[KDim], dtype=float_type),
                ts.FieldType(dims=[KDim], dtype=float_type),
            ],
            {},
            [
                r"Expected argument 'a' to be of type 'Field\[\[K\], int64\]', got 'Field\[\[K\], float64\]'",
                r"Expected argument 'b' to be of type 'Field\[\[K\], int64\]', got 'Field\[\[K\], float64\]'",
            ],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[IDim, JDim], dtype=int_type),
                ts.FieldType(dims=[KDim], dtype=int_type),
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [ts.FieldType(dims=[KDim], dtype=int_type), ts.FieldType(dims=[KDim], dtype=int_type)],
            {},
            [],
            ts.FieldType(dims=[KDim], dtype=float_type),
        ),
        (
            scanop_type,
            [
                ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                ts.FieldType(dims=[IDim, JDim], dtype=int_type),
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [
                ts.TupleType(
                    types=[
                        ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type),
                        ts.FieldType(dims=[IDim, JDim], dtype=int_type),
                    ]
                )
            ],
            {},
            [],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            tuple_scanop_type,
            [ts.TupleType(types=[ts.FieldType(dims=[IDim, JDim, KDim], dtype=int_type)])],
            {},
            [
                r"Expected argument 'a' to be of type 'tuple\[Field\[\[I, J, K\], int64\], "
                r"Field\[\[\.\.\.\], int64\]\]', got 'tuple\[Field\[\[I, J, K\], int64\]\]'."
            ],
            ts.FieldType(dims=[IDim, JDim, KDim], dtype=float_type),
        ),
        (
            ts.FunctionType(
                pos_only_args=[
                    ts_it.IteratorType(
                        position_dims="unknown", defined_dims=[], element_type=float_type
                    ),
                ],
                pos_or_kw_args={},
                kw_only_args={},
                returns=ts.VoidType(),
            ),
            [ts_it.IteratorType(position_dims=[IDim], defined_dims=[], element_type=float_type)],
            {},
            [],
            ts.VoidType(),
        ),
    ]


@pytest.mark.parametrize("symbol_type,expected", type_info_cases())
def test_type_info_basic(symbol_type, expected):
    for key in expected:
        assert getattr(type_info, key)(symbol_type) == expected[key]


def is_generic_cases() -> list[tuple[ts.TypeSpec, bool]]:
    deferred_type = ts.DeferredType(constraint=None)
    float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    concrete_field_type = ts.FieldType(dims=[TDim], dtype=float_type)

    def function_type(params: list[ts.TypeSpec]) -> ts.FunctionType:
        return ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={f"arg{i}": param for i, param in enumerate(params)},
            kw_only_args={},
            returns=ts.VoidType(),
        )

    return [
        (deferred_type, True),
        (float_type, False),
        (concrete_field_type, False),
        (ts.TupleType(types=[float_type, concrete_field_type]), False),
        # `DeferredType` nested inside a composite type, e.g. the program context signature
        #  of a scan operator with tuple arguments
        (ts.TupleType(types=[float_type, deferred_type]), True),
        (function_type([concrete_field_type]), False),
        (function_type([deferred_type]), True),
        (function_type([ts.TupleType(types=[deferred_type])]), True),
        (
            ts_ffront.ProgramType(definition=function_type([deferred_type])),
            True,
        ),
        (
            ts_ffront.FieldOperatorType(definition=function_type([concrete_field_type])),
            False,
        ),
    ]


@pytest.mark.parametrize("symbol_type,expected", is_generic_cases())
def test_is_generic(symbol_type: ts.TypeSpec, expected: bool):
    assert type_info.is_generic(symbol_type) == expected


float32_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
int32_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
float_var = ts.TypeVarType(name="T", constraints=(float32_type, float64_type))
mixed_var = ts.TypeVarType(name="U", constraints=(float64_type, int32_type))


class TestTypeVarType:
    def test_validation(self):
        with pytest.raises(ValueError, match="value-constrained"):
            ts.TypeVarType(name="T", constraints=())

    def test_identity_and_hashing(self):
        from gt4py.eve import utils as eve_utils

        same_var = ts.TypeVarType(name="T", constraints=(float32_type, float64_type))
        assert float_var == same_var
        assert hash(float_var) == hash(same_var)
        assert eve_utils.content_hash(float_var) == eve_utils.content_hash(same_var)
        assert float_var != ts.TypeVarType(name="S", constraints=(float32_type, float64_type))
        # constraint order is canonicalized, so it is not part of the identity
        assert float_var == ts.TypeVarType(name="T", constraints=(float64_type, float32_type))

    def test_is_generic(self):
        assert type_info.is_generic(float_var)
        assert type_info.is_generic(ts.FieldType(dims=[TDim], dtype=float_var))
        assert type_info.is_generic(
            ts.TupleType(types=[float64_type, ts.FieldType(dims=[TDim], dtype=float_var)])
        )

    @pytest.mark.parametrize(
        "predicate,var,expected",
        [
            (type_info.is_floating_point, float_var, True),
            (type_info.is_floating_point, mixed_var, False),
            (type_info.is_integral, float_var, False),
            (type_info.is_integral, ts.TypeVarType(name="I", constraints=(int32_type,)), True),
            (type_info.is_arithmetic, float_var, True),
            (type_info.is_arithmetic, mixed_var, True),
            (
                type_info.is_arithmetic,
                ts.TypeVarType(name="B", constraints=(bool_type, float64_type)),
                False,
            ),
            (type_info.is_logical, ts.TypeVarType(name="B", constraints=(bool_type,)), True),
            (type_info.is_logical, float_var, False),
            (type_info.is_arithmetic_scalar, float_var, True),
        ],
    )
    def test_predicates_evaluate_over_constraints(self, predicate, var, expected):
        assert predicate(var) == expected
        if predicate is not type_info.is_arithmetic_scalar:  # rejects fields by design
            assert predicate(ts.FieldType(dims=[TDim], dtype=var)) == expected

    def test_promote_same_var(self):
        assert type_info.promote(float_var, float_var) == float_var
        promoted = type_info.promote(
            ts.FieldType(dims=[TDim], dtype=float_var), ts.FieldType(dims=[TDim], dtype=float_var)
        )
        assert promoted == ts.FieldType(dims=[TDim], dtype=float_var)

    def test_promote_var_with_scalar_arg(self):
        promoted = type_info.promote(ts.FieldType(dims=[TDim], dtype=float_var), float_var)
        assert promoted == ts.FieldType(dims=[TDim], dtype=float_var)

    @pytest.mark.parametrize(
        "types",
        [
            (float_var, float64_type),
            (float_var, mixed_var),
            (ts.FieldType(dims=[TDim], dtype=float_var), float64_type),
            (
                ts.FieldType(dims=[TDim], dtype=float_var),
                ts.FieldType(dims=[TDim], dtype=float64_type),
            ),
        ],
    )
    def test_promote_mixing_error(self, types):
        with pytest.raises(ValueError, match="type variable"):
            type_info.promote(*types)


class TestBindTypeVars:
    def test_bind_from_field(self):
        binding = type_info.bind_type_vars(
            [ts.FieldType(dims=[TDim], dtype=float_var)],
            [ts.FieldType(dims=[TDim], dtype=float32_type)],
        )
        assert binding == {"T": float32_type}

    def test_bind_from_scalar_and_nested(self):
        binding = type_info.bind_type_vars(
            [ts.TupleType(types=[float_var, ts.FieldType(dims=[TDim], dtype=float_var)])],
            [ts.TupleType(types=[float64_type, ts.FieldType(dims=[TDim], dtype=float64_type)])],
        )
        assert binding == {"T": float64_type}

    def test_concrete_params_dont_bind(self):
        assert (
            type_info.bind_type_vars(
                [ts.FieldType(dims=[TDim], dtype=float64_type)],
                [ts.FieldType(dims=[TDim], dtype=float32_type)],
            )
            == {}
        )

    def test_inconsistent_binding(self):
        with pytest.raises(ValueError, match="bound inconsistently"):
            type_info.bind_type_vars(
                [
                    ts.FieldType(dims=[TDim], dtype=float_var),
                    ts.FieldType(dims=[TDim], dtype=float_var),
                ],
                [
                    ts.FieldType(dims=[TDim], dtype=float32_type),
                    ts.FieldType(dims=[TDim], dtype=float64_type),
                ],
            )

    def test_constraint_violation(self):
        with pytest.raises(ValueError, match="constraints"):
            type_info.bind_type_vars(
                [ts.FieldType(dims=[TDim], dtype=float_var)],
                [ts.FieldType(dims=[TDim], dtype=int32_type)],
            )


class TestSubstituteTypeVars:
    def test_substitute(self):
        generic = ts.TupleType(
            types=[float_var, ts.FieldType(dims=[TDim], dtype=float_var), int32_type]
        )
        substituted = type_info.substitute_type_vars(generic, {"T": float32_type})
        assert substituted == ts.TupleType(
            types=[float32_type, ts.FieldType(dims=[TDim], dtype=float32_type), int32_type]
        )
        assert not type_info.is_generic(substituted)

    def test_unbound_vars_are_kept(self):
        generic = ts.FieldType(dims=[TDim], dtype=float_var)
        assert type_info.substitute_type_vars(generic, {"S": float32_type}) == generic

    def test_concrete_is_returned_unchanged(self):
        concrete = ts.FieldType(dims=[TDim], dtype=float64_type)
        assert type_info.substitute_type_vars(concrete, {"T": float32_type}) is concrete

    def test_substitute_function_type(self):
        func_type = ts.FunctionType(
            pos_only_args=[ts.FieldType(dims=[TDim], dtype=float_var)],
            pos_or_kw_args={"a": float_var},
            kw_only_args={},
            returns=ts.FieldType(dims=[TDim], dtype=float_var),
        )
        substituted = type_info.substitute_type_vars(func_type, {"T": float64_type})
        assert substituted.pos_only_args[0] == ts.FieldType(dims=[TDim], dtype=float64_type)
        assert substituted.pos_or_kw_args["a"] == float64_type
        assert substituted.returns == ts.FieldType(dims=[TDim], dtype=float64_type)


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_accept_args(
    func_type: ts.TypeSpec,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
    expected: list,
    return_type: ts.TypeSpec,
):
    accepts_args = len(expected) == 0
    assert accepts_args == type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)

    if len(expected) > 0:
        with pytest.raises(ValueError) as exc_info:
            type_info.accepts_args(
                func_type, with_args=args, with_kwargs=kwargs, raise_exception=True
            )

        for expected_msg in expected:
            assert exc_info.match(expected_msg)


@pytest.mark.parametrize("func_type,args,kwargs,expected,return_type", callable_type_info_cases())
def test_return_type(
    func_type: ts.TypeSpec,
    args: list[ts.TypeSpec],
    kwargs: dict[str, ts.TypeSpec],
    expected: list,
    return_type: ts.TypeSpec,
):
    accepts_args = type_info.accepts_args(func_type, with_args=args, with_kwargs=kwargs)
    if accepts_args:
        assert type_info.return_type(func_type, with_args=args, with_kwargs=kwargs) == return_type


@pytest.mark.parametrize(
    "type_spec,expected",
    [
        (ts.ScalarType(kind=ts.ScalarKind.INT64), False),
        (
            ts.FieldType(dims=[Dimension("I")], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)),
            False,
        ),
        (
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT64),
                    ts.FieldType(
                        dims=[Dimension("I")], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
                    ),
                ]
            ),
            False,
        ),
        (
            ts.NamedCollectionType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT64),
                    ts.FieldType(
                        dims=[Dimension("I")], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
                    ),
                ],
                keys=["a", "b"],
                original_python_type="some.module:SomeClass",
            ),
            True,
        ),
        (
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT64),
                    ts.NamedCollectionType(
                        types=[
                            ts.ScalarType(kind=ts.ScalarKind.INT64),
                            ts.FieldType(
                                dims=[Dimension("I")],
                                dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                            ),
                        ],
                        keys=["x", "y"],
                        original_python_type="some.module:SomeClass",
                    ),
                ]
            ),
            True,
        ),
    ],
)
def test_needs_value_extraction(type_spec: ts.TypeSpec, expected: bool):
    assert type_info.needs_value_extraction(type_spec) is expected
