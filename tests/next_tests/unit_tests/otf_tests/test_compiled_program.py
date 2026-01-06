# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import eve, next as gtx
from gt4py.next import utils
from gt4py.next import errors, backend, broadcast, common
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import toolchain, arguments
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


@gtx.field_operator
def fop(cond: bool):
    return broadcast(cond, (TDim,))


@gtx.program
def prog(
    cond: bool,
    out: gtx.Field[gtx.Dims[TDim], bool],
):
    fop(cond, out=out)


def _verify_program_has_expected_true_value(program: itir.Program):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    assert isinstance(program.body[0].expr.args[0], itir.Literal)
    assert program.body[0].expr.args[0].value  # is True


def test_inlining_of_scalars_works():
    input_pair = toolchain.CompilableProgram(
        data=prog.definition_stage,
        args=arguments.CompileTimeArgs(
            args=list(prog.past_stage.past_node.type.definition.pos_or_kw_args.values()),
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
        out=gtx.zeros(domain={TDim: 1}, dtype=bool),
        offset_provider={},
    )

    _verify_program_has_expected_true_value(hijacked_program.data)


def _verify_program_has_expected_domain(
    program: itir.Program, expected_domain: gtx.Domain, uids: utils.IDGeneratorPool
):
    assert isinstance(program.body[0], itir.SetAt)
    assert isinstance(program.body[0].expr, itir.FunCall)
    assert program.body[0].expr.fun == itir.SymRef(id="fop")
    domain = CollapseTuple.apply(program.body[0].domain, within_stencil=False, uids=uids)
    assert domain == im.domain(common.GridType.CARTESIAN, expected_domain)


def test_inlining_of_static_domain_works(uids: utils.IDGeneratorPool):
    domain = gtx.Domain(dims=(TDim,), ranges=(gtx.UnitRange(0, 1),))
    input_pair = toolchain.CompilableProgram(
        data=prog.definition_stage,
        args=arguments.CompileTimeArgs(
            args=list(prog.past_stage.past_node.type.definition.pos_or_kw_args.values()),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={
                arguments.FieldDomainDescriptor: {"out": arguments.FieldDomainDescriptor(domain)}
            },
        ),
    )

    transformed = backend.DEFAULT_TRANSFORMS(input_pair).data
    _verify_program_has_expected_domain(transformed, domain, uids)
