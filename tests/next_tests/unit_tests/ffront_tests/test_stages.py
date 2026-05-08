# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront import stages


def _make_field_operator_definition(offset: int):
    def copy(a):
        return a + offset

    return copy


def _make_program_definition(offset: int):
    def copy_program(a, out):
        return a + out + offset

    return copy_program


def test_fingerprinter_hashes_functions_by_source_and_closure():
    first = _make_field_operator_definition(1)
    same = _make_field_operator_definition(1)
    different = _make_field_operator_definition(2)

    assert stages.fingerprinter(first) == stages.fingerprinter(same)
    assert stages.fingerprinter(first) != stages.fingerprinter(different)


def test_definition_stages_use_the_custom_fingerprinter():
    first_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(1))
    same_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(1))
    different_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(2))

    first_program = stages.DSLProgramDef(definition=_make_program_definition(1))
    same_program = stages.DSLProgramDef(definition=_make_program_definition(1))
    different_program = stages.DSLProgramDef(definition=_make_program_definition(2))

    assert first_fieldop.fingerprint == same_fieldop.fingerprint
    assert first_fieldop.fingerprint != different_fieldop.fingerprint
    assert first_program.fingerprint == same_program.fingerprint
    assert first_program.fingerprint != different_program.fingerprint
