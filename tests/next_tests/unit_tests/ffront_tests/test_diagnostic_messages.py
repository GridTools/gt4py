# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests pinning the rendered text of user-facing diagnostics.

These are deliberately end-to-end on the message level (in the spirit of
rustc's UI tests): they parse intentionally wrong programs and assert on the
rendered diagnostic, so that the quality of error messages cannot silently
regress. When changing a message, update the expectation here alongside.
"""

import re
import sys

import pytest

import gt4py.next as gtx
from gt4py.next import errors, float32, float64
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


IDim = gtx.Dimension("IDim")


def parse_error(func) -> errors.DSLError:
    with pytest.raises(errors.DSLError) as exc_info:
        FieldOperatorParser.apply_to_function(func)
    return exc_info.value


def test_undeclared_symbol_suggests_close_match():
    def misspelled(temperature: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        tmp_field = temperature * 2.0
        return tmp_feild  # noqa: F821 [undefined-name]

    err = parse_error(misspelled)

    assert isinstance(err, errors.UndefinedSymbolError)
    assert err.message == "Undeclared symbol 'tmp_feild'."
    assert err.hints == ["Did you mean 'tmp_field'?"]
    rendered = str(err)
    assert "return tmp_feild" in rendered
    assert re.search(r"\| +\^{9}", rendered), rendered
    assert "Hint: Did you mean 'tmp_field'?" in rendered


def test_undeclared_symbol_without_close_match_has_no_hint():
    def misspelled(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return completely_unrelated  # noqa: F821 [undefined-name]

    err = parse_error(misspelled)

    assert isinstance(err, errors.UndefinedSymbolError)
    assert err.hints == []


def test_while_loop_names_construct_and_alternative():
    def with_while(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        while True:
            a = a + 1.0
        return a

    err = parse_error(with_while)

    assert isinstance(err, errors.UnsupportedPythonFeatureError)
    assert err.message == "Unsupported Python syntax: 'while' loop."
    assert any("scan_operator" in hint for hint in err.hints)
    rendered = str(err)
    assert "while True:" in rendered
    assert "Note: Only a subset of Python is valid inside GT4Py functions." in rendered


def test_unlisted_construct_falls_back_to_ast_name():
    def with_string(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        f"{a}"
        return a

    err = parse_error(with_string)

    assert isinstance(err, errors.UnsupportedPythonFeatureError)
    assert "f-string" in err.message


def test_bool_field_arithmetic_suggests_where():
    def bool_arithmetic(
        a: gtx.Field[[IDim], float64], mask: gtx.Field[[IDim], bool]
    ) -> gtx.Field[[IDim], float64]:
        return a + mask

    err = parse_error(bool_arithmetic)

    assert err.label is not None and "'Field[[IDim], bool]'" in err.label
    assert err.related and "Field[[IDim], float64]" in err.related[0][1]
    assert any("where(mask, a, b)" in hint for hint in err.hints)
    rendered = str(err)
    # both operand labels are rendered into a single snippet of the offending line
    assert rendered.count("return a + mask") == 1
    assert re.search(r"\| +- the other operand has type", rendered), rendered


def test_dtype_mismatch_explains_promotion():
    def mixed_precision(
        a: gtx.Field[[IDim], float32], b: gtx.Field[[IDim], float64]
    ) -> gtx.Field[[IDim], float64]:
        return a + b

    err = parse_error(mixed_precision)

    assert err.notes == ["GT4Py does not implicitly convert between datatypes."]
    assert any("astype" in hint for hint in err.hints)
    assert len(err.related) == 2


def test_bool_op_suggests_bitwise_operators():
    def with_and(a: gtx.Field[[IDim], bool], b: gtx.Field[[IDim], bool]) -> gtx.Field[[IDim], bool]:
        return a and b

    err = parse_error(with_and)

    assert isinstance(err, errors.UnsupportedPythonFeatureError)
    assert any("'&' and '|'" in hint for hint in err.hints)


def test_add_note_uses_pep678_notes():
    # 'add_note' uses the standard PEP 678 mechanism ('__notes__'); the
    # structured 'notes' field is reserved for content authored at the raise
    # site, so the breadcrumb must not leak into it.
    err = errors.DSLError(None, "A message.")
    err.add_note("Extra context.")

    assert err.__notes__ == ["Extra context."]
    assert err.notes == []


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="On >=3.11 the traceback machinery renders '__notes__'; 'str()' does not.",
)
def test_add_note_folded_into_str_on_py310():
    err = errors.DSLError(None, "A message.")
    err.add_note("Extra context.")

    assert "Extra context." in str(err)


def test_toolchain_step_attaches_definition_context():
    from gt4py.next.ffront import stages as ffront_stages
    from gt4py.next.ffront.func_to_foast import func_to_foast

    def misspelled(temperature: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        tmp_field = temperature * 2.0
        return tmp_feild  # noqa: F821 [undefined-name]

    with pytest.raises(errors.DSLError) as exc_info:
        func_to_foast(ffront_stages.DSLFieldOperatorDef(definition=misspelled))

    assert "While processing the definition of 'misspelled'." in exc_info.value.__notes__


def test_global_statement_is_rejected_with_friendly_message():
    # 'global' used to crash an AST preprocessing pass with an AttributeError
    # because its 'names' field holds plain strings, not AST nodes.
    def with_global(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        global IDim
        return a

    err = parse_error(with_global)

    assert isinstance(err, errors.UnsupportedPythonFeatureError)
    assert err.message == "Unsupported Python syntax: 'global' statement."
    assert any("read-only" in hint for hint in err.hints)


def test_diagnostic_codes_are_stable():
    assert errors.UndefinedSymbolError.code == "undefined-symbol"
    assert errors.UnsupportedPythonFeatureError.code == "unsupported-syntax"
    assert errors.DSLError.code is None
