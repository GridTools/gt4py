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
from gt4py.next.otf import compiled_program, toolchain, arguments
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners import gtfn


def test_static_arg_from_enum():
    class SomeEnum(eve.IntEnum):
        FOO = 1

    static_arg = arguments.StaticArg(value=SomeEnum.FOO)
    assert static_arg.value == 1 and type(static_arg.value) is int


def test_static_arg_from_enum_tuple():
    class SomeEnum(eve.IntEnum):
        FOO = 1

    static_arg = arguments.StaticArg(value=(SomeEnum.FOO, SomeEnum.FOO))
    assert static_arg.value == (1, 1) and all(type(val) is int for val in static_arg.value)


def test_static_args_non_scalar_type():
    with pytest.raises(
        errors.DSLTypeError,
        match="only scalars.*can be static",
    ):
        static_arg = arguments.StaticArg(value=1)
        static_arg.validate(
            "foo", ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
        )


def test_sanitize_static_args_wrong_type():
    with pytest.raises(
        errors.DSLTypeError,
        match="expected 'int32'.*has.*'int64'",
    ):
        static_arg = arguments.StaticArg(value=gtx.int64(1))
        static_arg.validate("foo", ts.ScalarType(kind=ts.ScalarKind.INT32))


TDim = gtx.Dimension("TDim")


@pytest.fixture
def testee_prog():
    @gtx.field_operator
    def fop(cond: bool, a: gtx.Field[gtx.Dims[TDim], float], b: gtx.Field[gtx.Dims[TDim], float]):
        return a if cond else b

    @gtx.program(backend=gtfn.run_gtfn)
    def prog(
        cond: bool,
        a: gtx.Field[gtx.Dims[TDim], gtx.float64],
        b: gtx.Field[gtx.Dims[TDim], gtx.float64],
        out: gtx.Field[gtx.Dims[TDim], gtx.float64],
    ):
        fop(cond, a, b, out=out)

    return prog


def _verify_program_has_expected_true_value(program: itir.Program):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    assert isinstance(program.body[0].expr.args[0], itir.Literal)
    assert program.body[0].expr.args[0].value  # is True


def test_inlining_of_scalars_works(testee_prog):
    input_pair = toolchain.CompilableProgram(
        data=testee_prog.definition_stage,
        args=arguments.CompileTimeArgs(
            args=list(testee_prog.past_stage.past_node.type.definition.pos_or_kw_args.values()),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={
                arguments.StaticArg: {"cond": arguments.StaticArg(value=True)}
            },
        ),
    )

    transformed = backend.DEFAULT_TRANSFORMS(input_pair).data
    _verify_program_has_expected_true_value(transformed)


def test_inlining_of_scalar_works_integration(testee_prog):
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

    testee = testee_prog.with_backend(hacked_gtfn_backend).compile(cond=[True], offset_provider={})
    testee(
        cond=True,
        a=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        b=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        out=gtx.zeros(domain={TDim: 1}, dtype=gtx.float64),
        offset_provider={},
    )

    _verify_program_has_expected_true_value(hijacked_program.data)


def test_different_static_args_work_after_backend_change(testee_prog):
    prg1 = testee_prog.with_backend(gtfn.run_gtfn)
    prg2 = testee_prog.with_backend(gtfn.run_gtfn)

    # compile with static args
    prg1.compile(cond=[True], offset_provider={})

    # compile without static args
    prg2.compile(offset_provider={})


def test_different_static_args_work_after_static_params_change(testee_prog):
    testee_prog2 = testee_prog.with_compilation_options(static_params=["cond"])

    # compile without static args
    testee_prog.compile(offset_provider={})

    # compile with static args
    testee_prog2.compile(cond=[True], offset_provider={})


def test_different_static_args_break_same_prg_after_static_params_change(testee_prog):
    prg = testee_prog.with_compilation_options(static_params=[])

    # compile without static args
    prg.compile(offset_provider={})

    # compile with different static args
    with pytest.raises(
        ValueError,
        match="Argument descriptor StaticArg must be the same for all compiled programs",
    ):
        prg.compile(cond=[True], offset_provider={})
