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
from gt4py.next.ffront.func_to_past import ProgramParser


IDim = gtx.Dimension("IDim")


def parse_error(func) -> errors.DSLError:
    with pytest.raises(errors.DSLError) as exc_info:
        FieldOperatorParser.apply_to_function(func)
    return exc_info.value


def parse_program_error(func) -> errors.DSLError:
    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(func)
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


def test_numpy_style_attribute_on_field_is_a_dsl_error():
    # used to leak AttributeError: 'FieldType' object has no attribute 'T'
    def with_numpy_attr(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return a.T

    err = parse_error(with_numpy_attr)

    assert err.message == "Type 'Field[[IDim], float64]' has no attribute 'T'."
    assert any("NumPy-style" in note for note in err.notes)


def test_numpy_function_call_is_a_dsl_error():
    # used to leak ValueError: Type <class 'numpy.ufunc'> not supported
    import numpy as np

    def with_numpy_call(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return np.sin(a)

    err = parse_error(with_numpy_call)

    assert err.message == "'sin' cannot be used inside a GT4Py function."
    assert any("GT4Py built-in" in hint for hint in err.hints)


def test_missing_module_attribute_is_a_dsl_error():
    # used to leak AttributeError: module 'numpy' has no attribute 'sinn'
    import numpy as np

    def with_missing_attr(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return np.sinn(a)

    err = parse_error(with_missing_attr)

    assert "module 'numpy' has no attribute 'sinn'" in err.message


def test_absolute_field_index_is_a_dsl_error():
    # used to leak AttributeError: 'ScalarType' object has no attribute 'dim'
    def with_index(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return a[3]

    err = parse_error(with_index)

    assert err.message == "Fields cannot be indexed with 'int32'."
    assert any("field offset" in hint for hint in err.hints)


def test_tuple_index_out_of_range_is_a_dsl_error():
    # used to leak IndexError: list index out of range
    def with_oob_index(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        t = (a, a)
        return t[5]

    err = parse_error(with_oob_index)

    assert err.message == "Tuple index 5 is out of range."
    assert err.label == "this tuple has 2 elements"


def test_non_local_dimension_index_is_a_dsl_error():
    # used to crash with AssertionError on the dimension kind
    def with_dim_index(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return a[IDim(3)]

    err = parse_error(with_dim_index)

    assert "'IDim' is not a local (neighbor) dimension" in err.message


def test_unresolvable_string_annotation_is_a_dsl_error():
    # used to leak SyntaxError from typing.get_type_hints
    def with_bad_annotation(a: "not a type") -> gtx.Field[[IDim], float64]:  # noqa: F722 [syntax-error-in-forward-annotation]
        return a

    err = parse_error(with_bad_annotation)

    assert "Could not resolve type annotations of 'with_bad_annotation'" in err.message


def test_non_gt4py_parameter_annotation_is_a_dsl_error():
    # used to leak ValueError: Type <class 'list'> not supported
    def with_list_param(a: list) -> gtx.Field[[IDim], float64]:
        return a

    err = parse_error(with_list_param)

    assert isinstance(err, errors.InvalidParameterAnnotationError)
    assert any("GT4Py type" in hint for hint in err.hints)


def test_unresolvable_annotated_assignment_is_a_dsl_error():
    # used to leak NameError from eval'ing the annotation: 'gtx' is only
    # visible inside the function if it is also referenced in the body
    def with_ann_assign(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        b: gtx.Field[[IDim], float64] = a
        return b

    err = parse_error(with_ann_assign)

    assert "Invalid type annotation 'gtx.Field[[IDim], float64]'" in err.message
    assert any("GT4Py builtins" in note for note in err.notes)


@gtx.field_operator
def _copy_op(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
    return a


def test_expression_statement_in_program_is_a_dsl_error():
    # used to leak a TypeError from IR node validation
    def with_expr_stmt(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        a + a

    err = parse_program_error(with_expr_stmt)

    assert err.message == "Only calls to GT4Py operators are allowed as statements in a program."


def test_calling_program_from_program_is_a_dsl_error():
    # used to crash with AssertionError
    @gtx.program
    def inner(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        _copy_op(a, out=out)

    def with_program_call(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        inner(a, out)

    err = parse_program_error(with_program_call)

    assert err.message == "Program 'inner' cannot be called from within another program."


def test_plain_python_function_in_program_is_a_dsl_error():
    # used to leak ValueError: Invalid callable annotations ...
    def plain(a, out):
        return a

    def with_plain_call(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        plain(a, out=out)

    err = parse_program_error(with_plain_call)

    assert isinstance(err, errors.DSLTypeError)
    assert any("@field_operator" in hint for hint in err.hints)


def test_nested_tuple_unpacking_is_a_dsl_error():
    # used to leak AttributeError: 'TupleExpr' object has no attribute 'id'
    def with_nested_unpack(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        (b, c), d = (a, a), a
        return b

    err = parse_error(with_nested_unpack)

    assert err.message == "Nested tuple unpacking is not supported."


def test_invalid_literal_for_type_constructor_is_a_dsl_error():
    # used to leak ValueError at execution time, long after the definition
    from gt4py.next import int32

    def with_bad_cast(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        b = int32("abc")
        return a

    err = parse_error(with_bad_cast)

    assert "is not a valid literal for 'int32'" in err.message


def test_scan_operator_init_with_unsupported_type_is_a_dsl_error():
    # 'init=np.zeros(...)' used to crash with NumPy's ambiguous-truth-value
    # ValueError when fingerprinting the value
    import numpy as np

    KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)

    with pytest.raises(errors.DSLTypeError, match="Argument 'init'") as exc_info:

        @gtx.scan_operator(axis=KDim, forward=True, init=np.zeros(3))
        def scan_op(state: float64, x: float64) -> float64:
            return state + x

    assert "ndarray" in exc_info.value.message


def test_string_dimension_key_in_domain_is_a_dsl_error():
    # used to leak a TypeError from IR node validation
    def with_str_domain_key(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        _copy_op(a, out=out, domain={"IDim": (0, 10)})

    err = parse_program_error(with_str_domain_key)

    assert "Dictionary keys must be dimension objects" in err.message


@pytest.mark.parametrize(
    "feature, definition",
    [
        ("keyword-only parameters", "def f(a: F, *, b: F) -> F:\n    return a"),
        ("positional-only parameters", "def f(a: F, /) -> F:\n    return a"),
        ("'*args' parameters", "def f(*args: F) -> F:\n    return args[0]"),
        ("'**kwargs' parameters", "def f(a: F, **kw: F) -> F:\n    return a"),
        ("default values for parameters", "def f(a: F, b: float = 1.0) -> F:\n    return a"),
    ],
)
def test_unsupported_signature_features_are_dsl_errors(feature, definition, tmp_path):
    # these used to be silently dropped, leading to misleading
    # "Undeclared symbol" errors for the affected parameters
    import textwrap

    module = tmp_path / "sig_case.py"
    module.write_text(
        textwrap.dedent(
            """
            import gt4py.next as gtx
            from gt4py.next import float64
            IDim = gtx.Dimension("IDim")
            F = gtx.Field[[IDim], float64]
            """
        )
        + definition
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("sig_case", module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with pytest.raises(errors.UnsupportedPythonFeatureError) as exc_info:
        FieldOperatorParser.apply_to_function(mod.f)
    assert exc_info.value.message == f"Unsupported Python syntax: {feature}."


def test_out_of_range_integer_constant_explains_reason():
    def with_huge_constant(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        b = 99999999999999999999999999
        return a

    err = parse_error(with_huge_constant)

    assert err.message == "Invalid constant of type 'int'."
    assert any("out of range" in note for note in err.notes)


def test_calling_program_inside_field_operator_is_explained():
    # the message used to dump the entire 'ProgramType(...)' spec
    @gtx.program
    def some_prog(a: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        _copy_op(a, out=out)

    def with_prog_call(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return some_prog(a)

    err = parse_error(with_prog_call)

    assert err.message == "Programs cannot be called inside field operators."


# --- call-time diagnostics (embedded execution, no backend) ---------------

import numpy as np  # noqa: E402 [import-not-at-top-of-file]


def test_numpy_array_argument_in_direct_call_is_a_dsl_error():
    # used to crash an assert deep inside the embedded execution
    arr = np.zeros(5)
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLTypeError) as exc_info:
        _copy_op(arr, out=out, offset_provider={})

    assert "argument 1 has a type not supported by GT4Py: 'ndarray'" in exc_info.value.message
    assert any("as_field" in hint for hint in exc_info.value.hints)


def test_wrong_dims_argument_in_direct_call_is_a_dsl_error():
    # used to leak ValueError: Incompatible 'Domain' in assignment
    JDim = gtx.Dimension("JDim")
    a = gtx.as_field([JDim], np.zeros(5))
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLError, match="Invalid argument types in call to '_copy_op'"):
        _copy_op(a, out=out, offset_provider={})


def test_wrong_out_type_in_direct_call_is_a_dsl_error():
    # used to leak ValueError: Incompatible 'Domain' in assignment
    JDim = gtx.Dimension("JDim")
    a = gtx.as_field([IDim], np.zeros(5))
    out = gtx.as_field([JDim], np.zeros(5))

    with pytest.raises(errors.DSLTypeError, match="expected keyword argument 'out' to be of type"):
        _copy_op(a, out=out, offset_provider={})


def test_extra_argument_in_direct_call_is_a_dsl_error():
    # used to leak TypeError: ... takes 1 positional argument but 2 were given
    a = gtx.as_field([IDim], np.zeros(5))
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLError, match="Invalid argument types in call to '_copy_op'"):
        _copy_op(a, a, out=out, offset_provider={})


def test_missing_offset_provider_entry_is_a_dsl_error():
    # used to leak a KeyError
    Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))

    @gtx.field_operator
    def shift_op(f: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return f(Ioff[1])

    a = gtx.as_field([IDim], np.zeros(5))
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLError, match="Offset 'Ioff' not found") as exc_info:
        shift_op(a, out=out, offset_provider={})
    assert any("offset_provider" in hint for hint in exc_info.value.hints)


def test_invalid_domain_argument_is_a_dsl_error():
    # used to leak ValueError: '0' is not 'DomainLike'
    a = gtx.as_field([IDim], np.zeros(5))
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLTypeError, match="Invalid 'domain' argument"):
        _copy_op(a, out=out, domain=(0, 5), offset_provider={})


def test_numpy_array_argument_in_program_call_is_a_dsl_error():
    # used to leak ValueError: The truth value of an array ... is ambiguous
    @gtx.program
    def copy_prog(f: gtx.Field[[IDim], float64], out: gtx.Field[[IDim], float64]):
        _copy_op(f, out=out)

    arr = np.zeros(5)
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLTypeError) as exc_info:
        copy_prog(arr, out, offset_provider={})

    assert "argument 1 has a type not supported by GT4Py" in exc_info.value.message


def test_non_mapping_offset_provider_is_a_dsl_error():
    # used to be silently accepted until an offset lookup crashed (or not at all)
    a = gtx.as_field([IDim], np.zeros(5))
    out = gtx.as_field([IDim], np.zeros(5))

    with pytest.raises(errors.DSLTypeError, match="'offset_provider' must be a mapping"):
        _copy_op(a, out=out, offset_provider=[("Ioff", IDim)])


def test_unsupported_field_dtype_reports_dtype():
    # used to crash an assert in NdArrayField.from_array
    with pytest.raises(ValueError, match="unsupported dtype 'float16'"):
        gtx.as_field([IDim], np.zeros(5, dtype=np.float16))


def test_string_dimensions_in_as_field_are_rejected():
    # used to fail with a baffling "''D'' cannot be interpreted as 'UnitRange'"
    with pytest.raises(TypeError, match="must be 'Dimension' objects"):
        gtx.as_field(["IDim"], np.zeros(5))


def test_non_integral_neighbor_table_reports_dtype():
    # used to crash an assert in NdArrayConnectivityField.from_array
    Vertex = gtx.Dimension("Vertex")
    Edge = gtx.Dimension("Edge")
    V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)

    with pytest.raises(ValueError, match="integral dtype"):
        gtx.as_connectivity([Vertex, V2EDim], codomain=Edge, data=np.array([[0.5, 1.5]]))


def test_diagnostic_codes_are_stable():
    assert errors.UndefinedSymbolError.code == "undefined-symbol"
    assert errors.UnsupportedPythonFeatureError.code == "unsupported-syntax"
    assert errors.DSLError.code is None
