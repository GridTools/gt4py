# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next.ffront import stages
from gt4py.next.otf import arguments, recipes


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
        recipes.CompilableProgram(fieldop.definition_stage, arguments.CompileTimeArgs.empty())
    ).data
    samecode = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast(
        recipes.CompilableProgram(
            samecode_fieldop.definition_stage, arguments.CompileTimeArgs.empty()
        )
    ).data
    different = gtx.backend.DEFAULT_TRANSFORMS.func_to_foast(
        recipes.CompilableProgram(
            different_fieldop.definition_stage, arguments.CompileTimeArgs.empty()
        )
    ).data

    assert stages.fingerprint_stage(samecode) != stages.fingerprint_stage(foast)
    assert stages.fingerprint_stage(different) != stages.fingerprint_stage(foast)


def test_fingerprint_stage_program_def(program, samecode_program, different_program):
    assert stages.fingerprint_stage(samecode_program.definition_stage) != stages.fingerprint_stage(
        program.definition_stage
    )
    assert stages.fingerprint_stage(different_program.definition_stage) != stages.fingerprint_stage(
        program.definition_stage
    )


def test_fingerprint_stage_past_def(program, samecode_program, different_program):
    past = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        recipes.CompilableProgram(program.definition_stage, arguments.CompileTimeArgs.empty())
    )
    samecode = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        recipes.CompilableProgram(
            samecode_program.definition_stage, arguments.CompileTimeArgs.empty()
        )
    )
    different = gtx.backend.DEFAULT_TRANSFORMS.func_to_past(
        recipes.CompilableProgram(
            different_program.definition_stage, arguments.CompileTimeArgs.empty()
        )
    )

    assert stages.fingerprint_stage(samecode) != stages.fingerprint_stage(past)
    assert stages.fingerprint_stage(different) != stages.fingerprint_stage(past)
