import math
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
from functional.ffront.fbuiltins import Dimension, Field, float64, int32, int64
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners import roundtrip

from ..iterator_tests.math_builtin_test_data import math_builtin_test_data


# TODO(tehrengruber): extend to gtfn backend when the builtins are supported
fieldview_backend = roundtrip.executor

IDim = Dimension("IDim")

# TODO(tehrengruber): add tests for scalar arguments to builtin. To avoid code
#  bloat this is postponed until programatically creating field operators
#  becomes easier.


def make_builtin_field_operator(builtin_name: str):
    # TODO(tehrengruber): creating a field operator programmatically should be
    #  easier than what we need to do here.
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
    else:
        raise AssertionError(f"Unknown builtin `{builtin_name}`")

    closure_vars = {"IDim": IDim, builtin_name: getattr(fbuiltins, builtin_name)}

    loc = foast.SourceLocation(line=1, column=1, source="none")

    params = [
        foast.Symbol(id=k, type=symbol_makers.make_symbol_type_from_typing(type), location=loc)
        for k, type in annotations.items()
        if k != "return"
    ]
    args = [foast.Name(id=k, location=loc) for k, _ in annotations.items() if k != "return"]

    closure_var_symbols = [
        foast.Symbol(
            id=name,
            type=symbol_makers.make_symbol_type_from_value(val),
            namespace=ct.Namespace.CLOSURE,
            location=loc,
        )
        for name, val in closure_vars.items()
    ]

    foast_node = foast.FieldOperator(
        id=builtin_name + "_field_operator",
        definition=foast.FunctionDefinition(
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
            closure_vars=closure_var_symbols,
            params=params,
            location=loc,
        ),
        location=loc,
    )
    typed_foast_node = FieldOperatorTypeDeduction.apply(foast_node)

    return FieldOperator(
        foast_node=typed_foast_node,
        closure_vars=closure_vars,
        backend=fieldview_backend,
        definition=None,
    )


@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins_execution(builtin_name: str, inputs):
    if builtin_name == "gamma":
        # numpy has no gamma function
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    inps = [np_as_located_field(IDim)(np.asarray(input)) for input in inputs]
    expected = ref_impl(*inputs)
    out = np_as_located_field(IDim)(np.zeros_like(expected))

    builtin_field_op = make_builtin_field_operator(builtin_name)

    builtin_field_op(*inps, out=out, offset_provider={})

    assert np.allclose(np.asarray(out), expected)
