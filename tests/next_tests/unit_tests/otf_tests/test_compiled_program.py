# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import eve, next as gtx
from gt4py.next import errors, backend
from gt4py.next.ffront import type_specifications as ts_ffront
from gt4py.next.otf import compiled_program, toolchain, arguments
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import simple_mesh


def test_offset_provider_to_type_unsafe():
    mesh = simple_mesh(None)
    offset_provider = mesh.offset_provider

    compiled_program._offset_provider_to_type_unsafe_impl.cache_clear()

    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1

    offset_provider2 = {"V2E": mesh.offset_provider["V2E"]}
    compiled_program._offset_provider_to_type_unsafe(offset_provider2)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 2
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1


class SomeEnum(eve.IntEnum):
    FOO = 1


@pytest.mark.parametrize(
    "value, type_, expected",
    [
        (gtx.int32(1), ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (gtx.int64(1), ts.ScalarType(kind=ts.ScalarKind.INT64), gtx.int64(1)),
        (1, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (True, ts.ScalarType(kind=ts.ScalarKind.BOOL), True),
        (False, ts.ScalarType(kind=ts.ScalarKind.BOOL), False),
        (SomeEnum.FOO, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (
            (1, (2.0, gtx.float32(3.0))),
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT32),
                    ts.TupleType(
                        types=[
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
                        ]
                    ),
                ]
            ),
            (gtx.float32(1), (gtx.float64(2.0), gtx.float32(3.0))),
        ),
    ],
)
def test_sanitize_static_args(value, type_, expected):
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={
                "testee": type_,
            },
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )

    result = compiled_program._sanitize_static_args(
        "testee_program", {"testee": value}, program_type
    )
    assert result == {"testee": expected}


def test_sanitize_static_args_non_scalar_type():
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={
                "foo": ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
            },
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )
    with pytest.raises(
        errors.DSLTypeError,
        match="foo.*cannot be static",
    ):
        compiled_program._sanitize_static_args("foo_program", {"foo": gtx.int32(1)}, program_type)


def test_sanitize_static_args_wrong_type():
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"foo": ts.ScalarType(kind=ts.ScalarKind.INT32)},
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )
    with pytest.raises(errors.DSLTypeError, match="got 'int64'"):
        compiled_program._sanitize_static_args("foo_program", {"foo": gtx.int64(1)}, program_type)


def test_sanitize_static_args_non_existing_parameter():
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"foo": ts.ScalarType(kind=ts.ScalarKind.INT32)},
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )
    with pytest.raises(errors.DSLTypeError, match="'unknown_param'"):
        compiled_program._sanitize_static_args(
            "foo_program", {"unknown_param": gtx.int64(1)}, program_type
        )


TDim = gtx.Dimension("TDim")


@gtx.field_operator
def fop(cond: bool, a: gtx.Field[gtx.Dims[TDim], float], b: gtx.Field[gtx.Dims[TDim], float]):
    return a if cond else b


@gtx.program
def prog(
    cond: bool,
    a: gtx.Field[gtx.Dims[TDim], gtx.float64],
    b: gtx.Field[gtx.Dims[TDim], gtx.float64],
    out: gtx.Field[gtx.Dims[TDim], gtx.float64],
):
    fop(cond, a, b, out=out)


def _verify_program_has_expected_true_value(program: itir.Program):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    assert isinstance(program.body[0].expr.args[0], itir.Literal)
    assert program.body[0].expr.args[0].value  # is True


def test_inlining_of_scalars_works():
    args = prog.past_stage.past_node.type.definition.pos_or_kw_args
    args = [arguments.StaticArg(value=True, type_=v) if k == "cond" else v for k, v in args.items()]

    input_pair = toolchain.CompilableProgram(
        data=prog.definition_stage,
        args=arguments.CompileTimeArgs(args=args, kwargs={}, offset_provider={}, column_axis=None),
    )

    transformed = backend.DEFAULT_TRANSFORMS(input_pair).data
    _verify_program_has_expected_true_value(transformed)


def test_inlining_of_scalar_works_integration():
    """
    Test that `.compile` replaces the scalar arg in the program.
    Unlike the previous test, this test uses a full backend and makes sure the replacement step is there.
    """
    # Note: this is more an integration test, but does not execute a program, like the other integration tests.

    hijacked_program = None

    def pirate(program: toolchain.CompilableProgram):
        # Replaces the gtfn otf_workflow: and steals the compilable program,
        # then returns a dummy "CompiledProgram" that does nothing.
        nonlocal hijacked_program
        hijacked_program = program
        return lambda *args, **kwargs: None

    hacked_gtfn_backend = gtfn.GTFNBackendFactory(name_postfix="_custom", otf_workflow=pirate)

    testee = prog.with_backend(hacked_gtfn_backend).compile(cond=[True], offset_provider={})
    testee(
        cond=True,
        a=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        b=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        out=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        offset_provider={},
    )

    _verify_program_has_expected_true_value(hijacked_program.data)
