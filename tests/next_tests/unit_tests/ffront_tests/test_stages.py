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


def _make_field_operator_definition_elsewhere(offset: int):
    # Byte-identical body to `_make_field_operator_definition`, but defined at a
    # different source location, so the resulting functions must fingerprint
    # differently (location-sensitive frontend-stage keys).
    def copy(a):
        return a + offset

    return copy


def test_fingerprinter_hashes_functions_by_source_and_closure():
    first = _make_field_operator_definition(1)
    same = _make_field_operator_definition(1)
    different = _make_field_operator_definition(2)

    assert stages.fingerprinter(first) == stages.fingerprinter(same)
    assert stages.fingerprinter(first) != stages.fingerprinter(different)


def test_fingerprinter_is_location_sensitive():
    # Two functions with byte-identical source and closure but different source
    # locations must fingerprint differently, otherwise a cached lowering would
    # carry the wrong `SourceLocation`s (mislabeled errors / debug info).
    here = _make_field_operator_definition(1)
    elsewhere = _make_field_operator_definition_elsewhere(1)

    assert stages.fingerprinter(here) != stages.fingerprinter(elsewhere)


def test_definition_stages_use_the_custom_fingerprinter():
    first_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(1))
    same_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(1))
    different_fieldop = stages.DSLFieldOperatorDef(definition=_make_field_operator_definition(2))

    first_program = stages.DSLProgramDef(definition=_make_program_definition(1))
    same_program = stages.DSLProgramDef(definition=_make_program_definition(1))
    different_program = stages.DSLProgramDef(definition=_make_program_definition(2))

    assert stages.fingerprinter(first_fieldop) == stages.fingerprinter(same_fieldop)
    assert stages.fingerprinter(first_fieldop) != stages.fingerprinter(different_fieldop)
    assert stages.fingerprinter(first_program) == stages.fingerprinter(same_program)
    assert stages.fingerprinter(first_program) != stages.fingerprinter(different_program)
