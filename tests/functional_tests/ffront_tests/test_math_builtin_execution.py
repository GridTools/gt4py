from typing import Callable

import numpy as np
import pytest

from functional.ffront import (
    common_types as ct,
    fbuiltins,
    field_operator_ast as foast,
    symbol_makers,
)
from functional.ffront.decorator import FieldOperator
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    float64,
    int32,
    int64,
)
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.ffront.source_utils import CapturedVars
from functional.iterator.embedded import np_as_located_field

IDim = Dimension("IDim")

def make_builtin_field_operator(builtin_name: str):
    # construct annotation dictionary containing the input argument and return
    #  types
    if builtin_name in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES:
        annotations = {"inp": Field[[IDim], float64], "return": Field[[IDim], float64]}
    elif builtin_name in fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES:
        annotations = {"inp": Field[[IDim], float64], "return": Field[[IDim], float64]}
    elif builtin_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
        annotations = {"inp": Field[[IDim], float64], "return": Field[[IDim], bool]}
    elif builtin_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
        annotations = {
            "inp1": Field[[IDim], float64],
            "inp2": Field[[IDim], float64],
            "return": Field[[IDim], float64],
        }
    elif builtin_name in fbuiltins.BINARY_MATH_INT_BUILTIN_NAMES:
        annotations = {
            "inp1": Field[[IDim], int64],
            "inp2": Field[[IDim], int64],
            "return": Field[[IDim], int64],
        }
    else:
        raise AssertionError(f"Unknown builtin `{builtin_name}`")

    captured_vars = CapturedVars(
        nonlocals={"IDim": IDim},
        globals={builtin_name: getattr(fbuiltins, builtin_name)},
        annotations=annotations,
        builtins=set(),
        unbound=set(),
    )

    loc = foast.SourceLocation(line=1, column=1, source="none")

    params = [
        foast.Symbol(id=k, type=symbol_makers.make_symbol_type_from_typing(type), location=loc)
        for k, type in annotations.items()
        if k != "return"
    ]
    args = [foast.Name(id=k, location=loc) for k, _ in annotations.items() if k != "return"]

    captured_vars_nodes = [
        foast.Symbol(
            id=name,
            type=symbol_makers.make_symbol_type_from_value(val),
            namespace=ct.Namespace.CLOSURE,
            location=loc,
        )
        for name, val in {**captured_vars.globals, **captured_vars.nonlocals}.items()
    ]

    foast_node = foast.FieldOperator(
        id=builtin_name + "_field_operator",
        body=[
            foast.Return(
                value=foast.Call(
                    func=foast.Name(id=builtin_name, location=loc),
                    args=args,
                    kwargs={},
                    location=loc,
                ),
                location=loc,
            )
        ],
        captured_vars=captured_vars_nodes,
        params=params,
        location=loc,
    )
    typed_foast_node = FieldOperatorTypeDeduction.apply(foast_node)

    return FieldOperator(
        foast_node=typed_foast_node,
        captured_vars=captured_vars,
        externals={},
        backend="roundtrip",
        definition=None,
    )


def math_builtin_test_cases():
    return [
        # FIXME(ben): what about pow?
        # FIXME(ben): dataset is missing invalid ranges (mostly nan outputs)
        # FIXME(ben): we're not properly testing different datatypes
        # builtin name, tuple of arguments
        (
            "abs",
            ([-1, 1, -1.0, 1.0, 0, -0, 0.0, -0.0],),
        ),
        (
            "minimum",
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [
                    2,
                    2.0,
                    3.0,
                    2.0,
                    3,
                    2,
                    -2,
                    -2.0,
                    -3.0,
                    -2.0,
                    -3,
                    -2,
                ],
            ),
        ),
        (
            "maximum",
            (
                [2, 2.0, 2.0, 3.0, 2, 3, -2, -2.0, -2.0, -3.0, -2, -3],
                [
                    2,
                    2.0,
                    3.0,
                    2.0,
                    3,
                    2,
                    -2,
                    -2.0,
                    -3.0,
                    -2.0,
                    -3,
                    -2,
                ],
            ),
        ),
        (
            "mod",
            # ([6, 6.0, -6, 6.0, 7, -7.0, 4.8, 4], [2, 2.0, 2.0, -2, 3.0, -3, 1.2, -1.2]),
            ([6, 7, 4], [2, -2, -3]),
        ),
        (
            "sin",
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            "cos",
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            "tan",
            ([0, 0.1, -0.01, np.pi, -2.0 / 3.0 * np.pi, 2.0 * np.pi, 3, 1000, -1000],),
        ),
        (
            "arcsin",
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            "arccos",
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            "arctan",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "sinh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "cosh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "tanh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "arcsinh",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "arccosh",
            ([1, 1.0, 1.2, 1.7, 2, 2.0, 100, 103.7, 1000, 1379.89],),
        ),
        (
            "arctanh",
            ([-1.0, -1, -0.7, -0.2, -0.0, 0, 0.0, 0.2, 0.7, 1, 1.0],),
        ),
        (
            "sqrt",
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        (
            "exp",
            (
                [
                    -1002.3,
                    -1000,
                    -103.7,
                    -100,
                    -1.2,
                    -1.0,
                    -0.7,
                    -0.1,
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.7,
                    1.0,
                    1.2,
                    100,
                    103.7,
                    1000,
                    1002.3,
                ],
            ),
        ),
        (
            "log",
            (
                [
                    -0.0,
                    0,
                    0.0,
                    0.1,
                    0.9,
                    1,
                    1.0,
                    2.3,
                    4,
                    4.0,
                    16,
                    16.0,
                    34.7,
                    100,
                    100.0,
                    1000,
                    1337.1337,
                ],
            ),
        ),
        # (
        #    math_fun_field_op_gamma,
        #    np.frompyfunc(math.gamma, nin=1, nout=1),
        #    # FIXME(ben): math.gamma throws when it overflows, maybe should instead yield `np.inf`?
        #    # overflows very quickly, already at `173`
        #    ([-1002.3, -103.7, -1.2, -0.7, -0.1, 0.1, 0.7, 1.0, 1, 1.2, 100, 103.7, 170.5],),
        # ),
        (
            "cbrt",
            (
                [
                    -1003.2,
                    -704.3,
                    -100.5,
                    -10.4,
                    -1.5,
                    -1.001,
                    -0.7,
                    -0.01,
                    -0.0,
                    0.0,
                    0.01,
                    0.7,
                    1.001,
                    1.5,
                    10.4,
                    100.5,
                    704.3,
                    1003.2,
                ],
            ),
        ),
        (
            "isfinite",
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            "isinf",
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            "isnan",
            # TODO(BenWeber42): would be good to ensure we have nans with different bit patterns
            ([1000, 0, 1, np.pi, -np.inf, np.inf, np.nan, np.nan + 1],),
        ),
        (
            "floor",
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
        (
            "ceil",
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
        (
            "trunc",
            ([-3.4, -1.5, -0.6, -0.1, -0.0, 0.0, 0.1, 0.6, 1.5, 3.4],),
        ),
    ]

# FIXME(ben): this is a code clone from `./tests/functional_tests/iterator_tests/test_builtins.py`
# we should probably put that dataset somewhere so we can resue it for these tests
@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_cases())
def test_math_function_builtins_execution(builtin_name: str, inputs):
    ref_impl: Callable = getattr(np, builtin_name)
    inps = [np_as_located_field(IDim)(np.asarray(input)) for input in inputs]
    expected = ref_impl(*inputs)
    out = np_as_located_field(IDim)(np.zeros_like(expected))

    builtin_field_op = make_builtin_field_operator(builtin_name)

    builtin_field_op(*inps, out=out, offset_provider={})

    assert np.allclose(np.asarray(out), expected)
