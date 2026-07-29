# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront import dialect_ast_enums, field_operator_ast as foast


def compute_assign_indices(
    targets: list[foast.FieldSymbol | foast.TupleSymbol | foast.ScalarSymbol | foast.Starred],
    num_elts: int,
) -> list[tuple[int, int] | int]:
    """
    Compute a list of indices, mapping each target to its respective value(s).

    This function is used when mapping a tuple of targets to a tuple of values, and also handles the
    case in which tuple unpacking is done using a Starred operator.

    Examples
    --------
    The below are examples of different types of unpacking and the generated indices. Note that the
    indices in the tuple correspond to the lower and upper slice indices of a Starred variable.

    ..  code-block:: python

        # example pseudocode
        *a, b, c = (1, 2, 3, 4)
        [(0, 2), 2, 3]

        a, *b, c = (1, 2, 3, 4)
        (0, (1, 3), 3)

        a, b, *c = (1, 2, 3, 4)
        [0, 1, (2, 4)]
    """
    starred_lower, starred_upper = None, None
    for idx, elt in enumerate(targets):
        if isinstance(elt, foast.Starred):
            starred_lower = idx
            break
    for idx, elt in zip(reversed(range(num_elts)), reversed(targets)):
        if isinstance(elt, foast.Starred):
            starred_upper = idx + 1
            break
    if starred_lower is not None and starred_upper is not None:
        return [
            *list(range(0, starred_lower)),
            (starred_lower, starred_upper),
            *list(range(starred_upper, num_elts)),
        ]
    return list(range(0, num_elts))  # no starred target


def expr_to_index(expr: foast.Expr) -> int:
    """
    Convert an expression that represent an (integral) index to an int index.

    Args:
        expr: The expression to convert. Supported are literals of the form `+/-<int>`.

    Returns:
        The integer value of the index.

    Raises:
        ValueError: If the expression is not a valid index expression.
    """
    if isinstance(expr, foast.Constant):
        return expr.value
    if (
        isinstance(expr, foast.UnaryOp)
        and isinstance(expr.op, dialect_ast_enums.UnaryOperator)
        and isinstance(expr.operand, foast.Constant)
    ):
        if expr.op is dialect_ast_enums.UnaryOperator.USUB:
            return -expr.operand.value
        if expr.op is dialect_ast_enums.UnaryOperator.UADD:
            return expr.operand.value

    raise ValueError(f"Not an index: '{expr}'.")
