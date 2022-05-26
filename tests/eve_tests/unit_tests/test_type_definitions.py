# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


from __future__ import annotations

import pydantic
import pytest

import eve


def test_sentinel():
    from eve.type_definitions import NOTHING

    values = [0, 1, 2, NOTHING, 4, 6]

    assert values.index(NOTHING) == 3
    assert values[values.index(NOTHING)] is NOTHING


def test_symbol_types():
    from eve.type_definitions import SymbolName

    assert SymbolName("valid_name_01A") == "valid_name_01A"
    assert SymbolName.from_string("valid_name_01A") == "valid_name_01A"
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("$name_01A")
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("0name_01A")
    with pytest.raises(ValueError, match="string does not match regex"):
        SymbolName.from_string("name_01A ")

    LettersOnlySymbol = SymbolName.constrained(r"[a-zA-Z]+$")
    assert LettersOnlySymbol.from_string("validNAME") == "validNAME"
    with pytest.raises(ValueError, match="string does not match regex"):
        LettersOnlySymbol.from_string("name_a")
    with pytest.raises(ValueError, match="string does not match regex"):
        LettersOnlySymbol.from_string("name01")


class TestSourceLocation:
    def test_valid_position(self):
        eve.type_definitions.SourceLocation(line=1, column=1, source="source")

    def test_invalid_position(self):
        with pytest.raises(pydantic.ValidationError):
            eve.type_definitions.SourceLocation(line=1, column=-1, source="source")

    def test_str(self):
        loc = eve.type_definitions.SourceLocation(line=1, column=1, source="source")
        assert str(loc) == "<source: Line 1, Col 1>"
