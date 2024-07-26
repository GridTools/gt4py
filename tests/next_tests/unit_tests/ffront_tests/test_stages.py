# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses

import pytest

from gt4py import next as gtx
from gt4py.next.ffront import stages
from gt4py.next.otf import arguments, workflow


@pytest.fixture
def idim():
    yield gtx.Dimension("I")


@pytest.fixture
def jdim():
    yield gtx.Dimension("J")


@pytest.fixture
def fieldop(idim):
    @gtx.field_operator
    def copy(a: gtx.Field[[idim], gtx.int32]) -> gtx.Field[[idim], gtx.int32]:
        return a

    yield copy


@pytest.fixture
def samecode_fieldop(idim):
    @gtx.field_operator
    def copy(a: gtx.Field[[idim], gtx.int32]) -> gtx.Field[[idim], gtx.int32]:
        return a

    yield copy


@pytest.fixture
def different_fieldop(jdim):
    @gtx.field_operator
    def copy(a: gtx.Field[[jdim], gtx.int32]) -> gtx.Field[[jdim], gtx.int32]:
        return a

    yield copy


@pytest.fixture
def program(fieldop, idim):
    copy = fieldop

    @gtx.program
    def copy_program(a: gtx.Field[[idim], gtx.int32], out: gtx.Field[[idim], gtx.int32]):
        copy(a, out=out)

    yield copy_program


@pytest.fixture
def samecode_program(samecode_fieldop, idim):
    copy = samecode_fieldop

    @gtx.program
    def copy_program(a: gtx.Field[[idim], gtx.int32], out: gtx.Field[[idim], gtx.int32]):
        copy(a, out=out)

    yield copy_program


@pytest.fixture
def different_program(different_fieldop, jdim):
    copy = different_fieldop

    @gtx.program
    def copy_program(a: gtx.Field[[jdim], gtx.int32], out: gtx.Field[[jdim], gtx.int32]):
        copy(a, out=out)

    yield copy_program


def test_fingerprint_stage_field_op_def(fieldop, samecode_fieldop, different_fieldop):
    assert stages.fingerprint_stage(samecode_fieldop.definition_stage) != stages.fingerprint_stage(
        fieldop.definition_stage
    )
    assert stages.fingerprint_stage(different_fieldop.definition_stage) != stages.fingerprint_stage(
        fieldop.definition_stage
    )


def test_fingerprint_stage_foast_op_def(fieldop, samecode_fieldop, different_fieldop):
    foast = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast(
        workflow.DataArgsPair(fieldop.definition_stage, arguments.CompileTimeArgs.empty())
    ).data
    samecode = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast(
        workflow.DataArgsPair(samecode_fieldop.definition_stage, arguments.CompileTimeArgs.empty())
    ).data
    different = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast(
        workflow.DataArgsPair(different_fieldop.definition_stage, arguments.CompileTimeArgs.empty())
    ).data

    assert stages.fingerprint_stage(samecode) != stages.fingerprint_stage(foast)
    assert stages.fingerprint_stage(different) != stages.fingerprint_stage(foast)


@dataclasses.dataclass(frozen=True)
class ToFoastClosure(workflow.NamedStepSequence):
    func_to_foast: workflow.Workflow = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast
    foast_to_closure: workflow.Workflow = dataclasses.field(
        default=gtx.backend.DEFAULT_TRANSFORMS.field_view_op_to_prog,
    )


def test_fingerprint_stage_foast_closure(fieldop, samecode_fieldop, different_fieldop, idim, jdim):
    toolchain = ToFoastClosure()
    foast_closure = toolchain(
        workflow.DataArgsPair(
            data=fieldop.definition_stage,
            args=arguments.JITArgs(
                args=(gtx.zeros({idim: 10}, gtx.int32),),
                kwargs={
                    "out": gtx.zeros({idim: 10}, gtx.int32),
                    "from_fieldop": fieldop,
                },
            ),
        ),
    )
    samecode = toolchain(
        workflow.DataArgsPair(
            data=samecode_fieldop.definition_stage,
            args=arguments.JITArgs(
                args=(gtx.zeros({idim: 10}, gtx.int32),),
                kwargs={
                    "out": gtx.zeros({idim: 10}, gtx.int32),
                    "from_fieldop": samecode_fieldop,
                },
            ),
        )
    )
    different = toolchain(
        workflow.DataArgsPair(
            data=different_fieldop.definition_stage,
            args=arguments.JITArgs(
                args=(gtx.zeros({jdim: 10}, gtx.int32),),
                kwargs={
                    "out": gtx.zeros({jdim: 10}, gtx.int32),
                    "from_fieldop": different_fieldop,
                },
            ),
        )
    )
    different_args = toolchain(
        workflow.DataArgsPair(
            data=fieldop.definition_stage,
            args=arguments.JITArgs(
                args=(gtx.zeros({idim: 11}, gtx.int32),),
                kwargs={
                    "out": gtx.zeros({idim: 11}, gtx.int32),
                    "from_fieldop": fieldop,
                },
            ),
        )
    )

    assert stages.fingerprint_stage(samecode) != stages.fingerprint_stage(foast_closure)
    assert stages.fingerprint_stage(different) != stages.fingerprint_stage(foast_closure)
    assert stages.fingerprint_stage(different_args) != stages.fingerprint_stage(foast_closure)


def test_fingerprint_stage_program_def(program, samecode_program, different_program):
    assert stages.fingerprint_stage(samecode_program.definition_stage) != stages.fingerprint_stage(
        program.definition_stage
    )
    assert stages.fingerprint_stage(different_program.definition_stage) != stages.fingerprint_stage(
        program.definition_stage
    )


def test_fingerprint_stage_past_def(program, samecode_program, different_program):
    past = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        workflow.DataArgsPair(program.definition_stage, arguments.CompileTimeArgs.empty())
    )
    samecode = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        workflow.DataArgsPair(samecode_program.definition_stage, arguments.CompileTimeArgs.empty())
    )
    different = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        workflow.DataArgsPair(different_program.definition_stage, arguments.CompileTimeArgs.empty())
    )

    assert stages.fingerprint_stage(samecode) != stages.fingerprint_stage(past)
    assert stages.fingerprint_stage(different) != stages.fingerprint_stage(past)
