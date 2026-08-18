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

import ast
import re
import types

import pytest

import gt4py.next as gtx
from gt4py.next import errors, float32, float64
from gt4py.next.ffront import dialect_parser
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


IDim = gtx.Dimension("IDim")

# A PEP 695 alias whose value raises when it is evaluated, standing in for the
# common case of a typo'd dtype ('np.foat64') inside an alias definition.
_empty_module = types.ModuleType("_empty_module")
type BrokenFieldAlias = gtx.Field[gtx.Dims[IDim], _empty_module.foat64]


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


def test_try_statement_names_construct():
    # TODO(egparedes): cover 'try: ... except <Type>: ...' here once it is diagnosed
    # correctly. 'try/finally' below is one of the only two shapes that reach the
    # catalogue; the far more common 'except ValueError:' form is intercepted by
    # closure-variable type deduction first (see the TODO in 'func_to_foast'), so this
    # test must not be read as covering 'try' statements in general.
    def with_try(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        try:
            a = a + 1.0
        finally:
            pass
        return a

    err = parse_error(with_try)

    assert isinstance(err, errors.UnsupportedPythonFeatureError)
    assert err.message == "Unsupported Python syntax: 'try' statement."
    assert any("Exception handling" in hint for hint in err.hints)


def test_try_star_statement_is_catalogued():
    # 'try*' cannot be reached through the frontend: it always names an exception
    # type in its 'except*' clause, and that name is rejected as an unsupported
    # closure variable before the AST is visited. Pin the catalogue entry itself,
    # so the construct is named correctly if it ever does surface.
    node = ast.parse("try:\n    pass\nexcept* ValueError:\n    pass").body[0]
    assert isinstance(node, ast.TryStar)

    feature, hints = dialect_parser._describe_unsupported_feature(node)

    assert feature == "'try*' statement"
    assert any("Exception handling" in hint for hint in hints)


@pytest.mark.skipif(
    not hasattr(ast, "TemplateStr"), reason="PEP 750 t-strings require Python >= 3.14."
)
def test_post_floor_construct_is_catalogued():
    # Constructs newer than the supported floor are registered by name, so the
    # catalogue can name them without breaking the import on older interpreters.
    node = ast.parse('t"{a}"', mode="eval").body

    feature, hints = dialect_parser._describe_unsupported_feature(node)

    assert feature == "t-string"
    assert any("cannot be computed" in hint for hint in hints)


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


def test_invalid_cartesian_offset_suggests_valid_offsets():
    def foo(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return a(IDim + 0.25)

    err = parse_error(foo)

    assert err.message == "Invalid offset '0.25' for a Cartesian shift of dimension 'IDim'."
    assert any("half-integer offset" in hint for hint in err.hints)
    rendered = str(err)
    assert "return a(IDim + 0.25)" in rendered
    assert "Hint:" in rendered


def test_add_note_uses_pep678_notes():
    # 'add_note' uses the standard PEP 678 mechanism ('__notes__'); the
    # structured 'notes' field is reserved for content authored at the raise
    # site, so the breadcrumb must not leak into it.
    err = errors.DSLError(None, "A message.")
    err.add_note("Extra context.")

    assert err.__notes__ == ["Extra context."]
    assert err.notes == []


def test_toolchain_step_attaches_definition_context():
    from gt4py.next.ffront import stages as ffront_stages
    from gt4py.next.ffront.func_to_foast import func_to_foast

    def misspelled(temperature: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        tmp_field = temperature * 2.0
        return tmp_feild  # noqa: F821 [undefined-name]

    with pytest.raises(errors.DSLError) as exc_info:
        func_to_foast(ffront_stages.DSLFieldOperatorDef(definition=misspelled))

    assert "While processing the definition of 'misspelled'." in exc_info.value.__notes__


def test_diagnostic_codes_are_stable():
    assert errors.UndefinedSymbolError.code == "undefined-symbol"
    assert errors.UnsupportedPythonFeatureError.code == "unsupported-syntax"
    assert errors.DSLError.code is None


def test_unsupported_parameter_annotation_is_located():
    def bad_param(a: list[float64]) -> gtx.Field[[IDim], float64]:
        return a

    err = parse_error(bad_param)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert err.code == "invalid-annotation"
    assert err.location is not None
    rendered = str(err)
    assert "a: list[float64]" in rendered
    assert "Hint:" in rendered


def test_unsupported_return_annotation_is_located():
    def bad_return(a: gtx.Field[[IDim], float64]) -> list[float64]:
        return a

    err = parse_error(bad_return)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert "return type annotation" in err.message
    # The span covers the annotation, not the whole function definition (which used
    # to span both lines). See the note on caret-run regexes in
    # 'test_unsupported_variable_annotation_is_located'.
    assert err.location.line == err.location.end_line
    assert err.location.end_column - err.location.column == len("list[float64]")
    assert re.search(r"\| +\^{13}(?!\^)", str(err)), str(err)


def test_return_type_mismatch_points_at_the_returned_value():
    def mismatch(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float32]:
        return a

    err = parse_error(mismatch)

    assert "does not match deduced return type" in err.message
    # The deduced type comes from the returned expression, so that is what is
    # underlined rather than the whole function definition.
    assert err.location.line == err.location.end_line
    assert err.location.end_column - err.location.column == len("a")
    assert re.search(r"\| +\^(?!\^)", str(err)), str(err)


def test_return_type_mismatch_with_several_returns_falls_back_to_the_function():
    def two_returns(a: gtx.Field[[IDim], float64], cond: bool) -> gtx.Field[[IDim], float32]:
        if cond:
            return a
        else:
            return a + a

    err = parse_error(two_returns)

    assert "does not match deduced return type" in err.message
    # No single expression the deduced type can be pinned on, so the whole function
    # definition stays the primary span rather than an arbitrarily picked return.
    assert err.location.line < err.location.end_line


def test_bad_return_annotation_is_reported_before_body_errors():
    # The return annotation is checked while parsing the signature, so it is reported
    # even when the body would also fail to type-deduce. This ordering changed when
    # the check moved out of the post-processing step to get a usable location.
    def both_wrong(a: gtx.Field[[IDim], float64]) -> list[float64]:
        return a + "not a field"

    err = parse_error(both_wrong)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert "return type annotation" in err.message


def test_unsupported_variable_annotation_is_located():
    def bad_var(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        tmp: list[float64] = a
        return tmp

    err = parse_error(bad_var)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert "variable type annotation" in err.message
    rendered = str(err)
    assert "tmp: list[float64]" in rendered
    # The span covers the annotation only, not the whole statement. Asserted on the
    # columns rather than only on the carets, because a caret-run regex without a
    # trailing lookahead also matches any *longer* run and so pins nothing.
    assert err.location.line == err.location.end_line
    assert err.location.end_column - err.location.column == len("list[float64]")
    assert re.search(r"\| +\^{13}(?!\^)", rendered), rendered


def test_valid_variable_annotation_is_accepted():
    # Regression test: the annotation of a valid declaration is turned into a
    # location-less 'ast.Constant' by 'StringifyAnnotationsPass', so asking for
    # its source location used to crash before the pass preserved it.
    def annotated(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        tmp: float64 = 1.0
        return a + tmp

    FieldOperatorParser.apply_to_function(annotated)


def test_invalid_annotation_keeps_the_underlying_reason_as_a_note():
    def mistyped(a: gtx.Field) -> gtx.Field[[IDim], float64]:
        return a

    err = parse_error(mistyped)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert any("Field type requires two arguments" in note for note in err.notes)


def test_broken_type_alias_annotation_is_located():
    # A PEP 695 alias body is only evaluated when the annotation is resolved, so a
    # typo inside it surfaces during parsing. It has to come out as a located
    # diagnostic naming the typo, not as a raw 'AttributeError' traceback.
    def broken(a: BrokenFieldAlias) -> gtx.Field[[IDim], float64]:
        return a

    err = parse_error(broken)

    assert isinstance(err, errors.InvalidAnnotationError)
    assert err.location is not None
    assert any("foat64" in note for note in err.notes)
    # The span covers the parameter, not the whole signature. It includes the
    # parameter name because 'visit_arg' locates the whole 'ast.arg'; only
    # 'visit_AnnAssign' narrows down to the annotation itself.
    assert err.location.line == err.location.end_line
    assert err.location.end_column - err.location.column == len("a: BrokenFieldAlias")
    assert re.search(r"\| +\^{19}(?!\^)", str(err)), str(err)
