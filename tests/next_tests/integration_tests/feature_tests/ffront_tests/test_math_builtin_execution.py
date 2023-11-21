# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import math
from typing import Callable

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.ffront import dialect_ast_enums, fbuiltins, field_operator_ast as foast
from gt4py.next.ffront.decorator import FieldOperator
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.type_system import type_translation

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case, unstructured_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
)
from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data


# TODO: reduce duplication with `test_math_unary_builtins`

# TODO(tehrengruber): add tests for scalar arguments to builtin. To avoid code
#  bloat this is postponed until programatically creating field operators
#  becomes easier.


def make_builtin_field_operator(builtin_name: str):
    # TODO(tehrengruber): creating a field operator programmatically should be
    #  easier than what we need to do here.
    # construct annotation dictionary containing the input argument and return
    #  types
    if builtin_name in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES:
        annotations = {"inp": cases.IFloatField, "return": cases.IFloatField}
    elif builtin_name in fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES:
        annotations = {"inp": cases.IFloatField, "return": cases.IFloatField}
    elif builtin_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
        annotations = {"inp": cases.IFloatField, "return": cases.IBoolField}
    elif builtin_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
        annotations = {
            "inp1": cases.IFloatField,
            "inp2": cases.IFloatField,
            "return": cases.IFloatField,
        }
    else:
        raise AssertionError(f"Unknown builtin `{builtin_name}`")

    closure_vars = {"IDim": IDim, builtin_name: getattr(fbuiltins, builtin_name)}

    loc = foast.SourceLocation(line=1, column=1, filename="none")

    params = [
        foast.Symbol(id=k, type=type_translation.from_type_hint(type), location=loc)
        for k, type in annotations.items()
        if k != "return"
    ]
    args = [foast.Name(id=k, location=loc) for k, _ in annotations.items() if k != "return"]

    closure_var_symbols = [
        foast.Symbol(
            id=name,
            type=type_translation.from_value(val),
            namespace=dialect_ast_enums.Namespace.CLOSURE,
            location=loc,
        )
        for name, val in closure_vars.items()
    ]

    foast_node = foast.FieldOperator(
        id=builtin_name + "_field_operator",
        definition=foast.FunctionDefinition(
            id=builtin_name + "_field_operator",
            body=foast.BlockStmt(
                stmts=[
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
                location=loc,
            ),
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
        backend=None,
        definition=None,
    )


@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins_execution(cartesian_case, builtin_name: str, inputs):
    if cartesian_case.backend is None:
        # TODO(havogt) find a way that works for embedded
        pytest.xfail("Test does not have a field view program.")
    if builtin_name == "gamma":
        # numpy has no gamma function
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    inps = [cartesian_case.as_field([IDim], np.asarray(input)) for input in inputs]
    expected = ref_impl(*inputs)
    out = cartesian_case.as_field([IDim], np.zeros_like(expected))

    builtin_field_op = make_builtin_field_operator(builtin_name).with_backend(
        cartesian_case.backend
    )

    builtin_field_op(*inps, out=out, offset_provider={})

    assert np.allclose(out.asnumpy(), expected)
