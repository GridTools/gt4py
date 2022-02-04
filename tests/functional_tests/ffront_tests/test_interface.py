"""Test field view interface.

Basic Interface Tests
=====================

    - declare a connectivity
    - create and run a stencil
        - field args declaration
        - scalar args
    - create and run a fencil
        - pass fields
        - pass connectivities (at run time, later at compile time too)
        - out field
        - think about ways to pass backend/connectivities etc
            (in function signature / in functor config method / with block)
    - built-in field operators
        - arithmetics
        - shift
        - neighbor reductions
        - math functions: abs(), max(), min, mod(), sin(), cos(), tan(), arcsin(), arccos(),
            arctan(), sqrt(), exp(), log(), isfinite(), isinf(), isnan(), floor(), ceil(), trunc()
    - evaluation test cases
"""
import pytest

import functional.ffront.field_operator_ast as foast
from functional.common import Field
from functional.ffront import common_types
from functional.ffront.fbuiltins import float32, float64, int64
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError
from functional.ffront.symbol_makers import FieldOperatorTypeError
from functional.iterator import ir as itir
from functional.iterator.builtins import (
    and_,
    deref,
    divides,
    eq,
    greater,
    if_,
    less,
    make_tuple,
    minus,
    multiplies,
    not_,
    or_,
    plus,
    tuple_get,
)


DEREF = itir.SymRef(id=deref.fun.__name__)
PLUS = itir.SymRef(id=plus.fun.__name__)
MINUS = itir.SymRef(id=minus.fun.__name__)
MULTIPLIES = itir.SymRef(id=multiplies.fun.__name__)
DIVIDES = itir.SymRef(id=divides.fun.__name__)
MAKE_TUPLE = itir.SymRef(id=make_tuple.fun.__name__)
TUPLE_GET = itir.SymRef(id=tuple_get.fun.__name__)
IF = itir.SymRef(id=if_.fun.__name__)
NOT = itir.SymRef(id=not_.fun.__name__)
GREATER = itir.SymRef(id=greater.fun.__name__)
LESS = itir.SymRef(id=less.fun.__name__)
EQ = itir.SymRef(id=eq.fun.__name__)
AND = itir.SymRef(id=and_.fun.__name__)
OR = itir.SymRef(id=or_.fun.__name__)


COPY_FUN_DEF = itir.FunctionDefinition(
    id="copy_field",
    params=[itir.Sym(id="inp")],
    expr=itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")]),
)


# --- Parsing ---
def test_invalid_syntax_error_empty_return():
    """Field operator syntax errors point to the file, line and column."""

    def wrong_syntax(inp: Field[..., "float64"]):
        return

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(
            r"Invalid Field Operator Syntax: "
            r"Empty return not allowed \(test_interface.py, line 81\)"
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(wrong_syntax)


def test_untyped_arg():
    """Field operator parameters must be type annotated."""

    def untyped(inp):
        return inp

    with pytest.raises(
        FieldOperatorSyntaxError,
        match="Untyped parameters not allowed!",
    ):
        _ = FieldOperatorParser.apply_to_function(untyped)


def test_mistyped_arg():
    """Field operator parameters must be type annotated."""

    def mistyped(inp: Field):
        return inp

    with pytest.raises(
        FieldOperatorTypeError,
        match="Field type requires two arguments, got 0!",
    ):
        _ = FieldOperatorParser.apply_to_function(mistyped)


def test_return_type():
    """Return type annotation should be stored on the FieldOperator."""

    def rettype(inp: Field[..., float64]) -> Field[..., float64]:
        return inp

    parsed = FieldOperatorParser.apply_to_function(rettype)

    assert parsed.body[-1].value.type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )


def test_invalid_syntax_no_return():
    """Field operators must end with a return statement."""

    def no_return(inp: Field[..., "float64"]):
        tmp = inp  # noqa

    with pytest.raises(
        FieldOperatorSyntaxError,
        match="Field operator must return a field expression on the last line!",
    ):
        _ = FieldOperatorParser.apply_to_function(no_return)


def test_invalid_assign_to_expr():
    """Assigning to subscripts disallowed until a usecase can be found."""

    def invalid_assign_to_expr(inp1: Field[..., "float64"], inp2: Field[..., "float64"]):
        tmp = inp1
        tmp[-1] = inp2
        return tmp

    with pytest.raises(FieldOperatorSyntaxError, match=r"Can only assign to names! \(.*\)"):
        _ = FieldOperatorParser.apply_to_function(invalid_assign_to_expr)


def test_copy_lower():
    def copy_field(inp: Field[..., "float64"]):
        return inp

    # ast_passes
    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)
    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_syntax_unpacking():
    """For now, only single target assigns are allowed."""

    def unpacking(inp1: Field[..., "float64"], inp2: Field[..., "float64"]):
        tmp1, tmp2 = inp1, inp2  # noqa
        return tmp1

    parsed = FieldOperatorParser.apply_to_function(unpacking)
    lowered = FieldOperatorLowering.apply(parsed)
    assert lowered.expr == itir.FunCall(
        fun=itir.SymRef(id="tuple_get"),
        args=[
            itir.FunCall(
                fun=MAKE_TUPLE,
                args=[
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp1")]),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp2")]),
                ],
            ),
            itir.IntLiteral(value=0),
        ],
    )


def test_temp_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp = inp
        inp = tmp
        tmp2 = inp
        return tmp2

    parsed = FieldOperatorParser.apply_to_function(copy_field)

    assert parsed.symtable_["tmp$0"].type == common_types.FieldType(
        dims=Ellipsis,
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )

    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_annotated_assignment():
    def copy_field(inp: Field[..., "float64"]):
        tmp: Field[..., "float64"] = inp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(copy_field)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered == COPY_FUN_DEF
    assert lowered.expr == COPY_FUN_DEF.expr


def test_clashing_annotated_assignment():
    def clashing(inp: Field[..., "float64"]):
        tmp: Field[..., "int64"] = inp
        return tmp

    with pytest.raises(FieldOperatorTypeDeductionError, match="type inconsistency"):
        _ = FieldOperatorParser.apply_to_function(clashing)


def test_call():
    def identity(x: Field[..., "float64"]) -> Field[..., "float64"]:
        return x

    def call(inp: Field[..., "float64"]) -> Field[..., "float64"]:
        return identity(inp)

    parsed = FieldOperatorParser.apply_to_function(call)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=itir.SymRef(id="identity"), args=[itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")])]
    )


def test_temp_tuple():
    """Returning a temp tuple should work."""

    def temp_tuple(a: Field[..., float64], b: Field[..., int64]):
        tmp = a, b
        return tmp

    parsed = FieldOperatorParser.apply_to_function(temp_tuple)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MAKE_TUPLE,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_unary_ops():
    def unary(inp: Field[..., "float64"]):
        tmp = +inp
        tmp = -tmp
        return tmp

    parsed = FieldOperatorParser.apply_to_function(unary)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MINUS,
        args=[
            itir.IntLiteral(value=0),
            itir.FunCall(
                fun=PLUS,
                args=[
                    itir.IntLiteral(value=0),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="inp")]),
                ],
            ),
        ],
    )


def test_unary_not():
    def unary_not(cond: Field[..., "bool"]):
        return not cond

    parsed = FieldOperatorParser.apply_to_function(unary_not)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=NOT, args=[itir.FunCall(fun=DEREF, args=[itir.SymRef(id="cond")])]
    )


def test_binary_plus():
    def plus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a + b

    parsed = FieldOperatorParser.apply_to_function(plus)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=PLUS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_mult():
    def mult(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a * b

    parsed = FieldOperatorParser.apply_to_function(mult)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MULTIPLIES,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_minus():
    def minus(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a - b

    parsed = FieldOperatorParser.apply_to_function(minus)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=MINUS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_div():
    def division(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a / b

    parsed = FieldOperatorParser.apply_to_function(division)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=DIVIDES,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_and():
    def bit_and(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a & b

    parsed = FieldOperatorParser.apply_to_function(bit_and)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=AND,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_or():
    def bit_or(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a | b

    parsed = FieldOperatorParser.apply_to_function(bit_or)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=OR,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_binary_pow():
    def power(inp: Field[..., "float64"]):
        return inp ** 3

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`\*\*` operator not supported!"),
    ):
        _ = FieldOperatorParser.apply_to_function(power)


def test_binary_mod():
    def power(inp: Field[..., "int64"]):
        return inp % 3

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`%` operator not supported!"),
    ):
        _ = FieldOperatorParser.apply_to_function(power)


def test_compare_gt():
    def comp_gt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a > b

    parsed = FieldOperatorParser.apply_to_function(comp_gt)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=GREATER,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_lt():
    def comp_lt(a: Field[..., "float64"], b: Field[..., "float64"]):
        return a < b

    parsed = FieldOperatorParser.apply_to_function(comp_lt)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=LESS,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_eq():
    def comp_eq(a: Field[..., "int64"], b: Field[..., "int64"]):
        return a == b

    parsed = FieldOperatorParser.apply_to_function(comp_eq)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=EQ,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
        ],
    )


def test_compare_chain():
    def compare_chain(a: Field[..., "float64"], b: Field[..., "float64"], c: Field[..., "float64"]):
        return a > b > c

    parsed = FieldOperatorParser.apply_to_function(compare_chain)
    lowered = FieldOperatorLowering.apply(parsed)

    assert lowered.expr == itir.FunCall(
        fun=GREATER,
        args=[
            itir.FunCall(fun=DEREF, args=[itir.SymRef(id="a")]),
            itir.FunCall(
                fun=GREATER,
                args=[
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="b")]),
                    itir.FunCall(fun=DEREF, args=[itir.SymRef(id="c")]),
                ],
            ),
        ],
    )


def test_bool_and():
    def bool_and(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a and b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`and` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_and)


def test_bool_or():
    def bool_or(a: Field[..., "bool"], b: Field[..., "bool"]):
        return a or b

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(r"`or` operator not allowed!"),
    ):
        _ = FieldOperatorParser.apply_to_function(bool_or)


# --- External symbols ---
def test_closure_symbols():
    import numpy as np

    nonlocal_unused = 0
    nonlocal_float = 2.3
    nonlocal_np_scalar = np.float32(3.4)

    def operator_with_refs(inp: Field[..., "float64"], inp2: Field[..., "float32"]):
        a = inp + nonlocal_float
        b = inp2 + nonlocal_np_scalar
        return a, b

    parsed = FieldOperatorParser.apply_to_function(operator_with_refs)
    assert parsed.symtable_["nonlocal_float"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT64, shape=None
    )
    assert parsed.symtable_["nonlocal_np_scalar"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT32, shape=None
    )
    assert "nonlocal_unused" not in parsed.symtable_


def test_external_symbols():
    import numpy as np

    def operator_with_externals(inp: Field[..., "float64"], inp2: Field[..., "float32"]):
        from __externals__ import ext_float, ext_np_scalar

        a = inp + ext_float
        b = inp2 + ext_np_scalar
        return a, b

    parsed = FieldOperatorParser.apply_to_function(
        operator_with_externals,
        externals=dict(ext_float=2.3, ext_np_scalar=np.float32(3.4), ext_unused=0),
    )
    assert parsed.symtable_["ext_float"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT64, shape=None
    )
    assert parsed.symtable_["ext_np_scalar"].type == common_types.ScalarType(
        kind=common_types.ScalarKind.FLOAT32, shape=None
    )
    assert "ext_unused" not in parsed.symtable_
