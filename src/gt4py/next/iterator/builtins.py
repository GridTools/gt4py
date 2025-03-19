# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.dispatcher import Dispatcher


builtin_dispatch = Dispatcher()


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


@builtin_dispatch
def as_fieldop(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def index(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def broadcast(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def can_deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def shift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def neighbors(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def map_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_const_list(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def list_get(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def lift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def reduce(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def scan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cartesian_domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def unstructured_domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def named_range(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def if_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cast_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def not_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def and_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def or_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def xor_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def minus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def plus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def multiplies(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def divides(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def floordiv(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def eq(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def less(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def less_equal(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater_equal(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def not_eq(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_tuple(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tuple_get(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def abs(*args):  # noqa: A001 [builtin-variable-shadowing]
    raise BackendNotSelectedError()


@builtin_dispatch
def sin(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cos(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arcsin(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arccos(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arctan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def sinh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cosh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def tanh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arcsinh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arccosh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def arctanh(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def sqrt(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def exp(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def log(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def gamma(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cbrt(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def floor(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def ceil(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def trunc(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def neg(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isfinite(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isinf(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def isnan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def minimum(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def maximum(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def fmod(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def mod(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def power(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def int(*args):  # noqa: A001 [builtin-variable-shadowing]
    raise BackendNotSelectedError()


@builtin_dispatch
def int8(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def uint8(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def int16(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def uint16(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def int32(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def uint32(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def int64(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def uint64(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def float(*args):  # noqa: A001 [builtin-variable-shadowing]
    raise BackendNotSelectedError()


@builtin_dispatch
def float32(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def float64(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def bool(*args):  # noqa: A001 [builtin-variable-shadowing]
    raise BackendNotSelectedError()


UNARY_MATH_NUMBER_BUILTINS = {"abs", "neg"}
UNARY_LOGICAL_BUILTINS = {"not_"}
UNARY_MATH_FP_BUILTINS = {
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sqrt",
    "exp",
    "log",
    "gamma",
    "cbrt",
    "floor",
    "ceil",
    "trunc",
}
UNARY_MATH_FP_PREDICATE_BUILTINS = {"isfinite", "isinf", "isnan"}
BINARY_MATH_NUMBER_BUILTINS = {
    "plus",
    "minus",
    "multiplies",
    "divides",
    "mod",
    "floordiv",  # TODO see https://github.com/GridTools/gt4py/issues/1136
    "minimum",
    "maximum",
    "fmod",
}
BINARY_MATH_COMPARISON_BUILTINS = {"eq", "less", "greater", "greater_equal", "less_equal", "not_eq"}
BINARY_LOGICAL_BUILTINS = {"and_", "or_", "xor_"}


#: builtin / dtype used to construct integer indices, like domain bounds
INTEGER_INDEX_BUILTIN = "int32"
INTEGER_TYPE_BUILTINS = {
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
}
FLOATING_POINT_TYPE_BUILTINS = {"float32", "float64"}
TYPE_BUILTINS = {*INTEGER_TYPE_BUILTINS, *FLOATING_POINT_TYPE_BUILTINS, "bool"}

ARITHMETIC_BUILTINS = {
    *UNARY_MATH_NUMBER_BUILTINS,
    *UNARY_LOGICAL_BUILTINS,
    *UNARY_MATH_FP_BUILTINS,
    *UNARY_MATH_FP_PREDICATE_BUILTINS,
    *BINARY_MATH_NUMBER_BUILTINS,
    "power",
    *BINARY_MATH_COMPARISON_BUILTINS,
    *BINARY_LOGICAL_BUILTINS,
}

BUILTINS = {
    "as_fieldop",  # `as_fieldop(stencil, domain)` creates field_operator from stencil (domain is optional, but for now required for embedded execution)
    "broadcast",
    "can_deref",
    "cartesian_domain",
    "cast_",
    "deref",
    "if_",
    "index",  # `index(dim)` creates a dim-field that has the current index at each point
    "shift",
    "list_get",
    "lift",
    "make_const_list",
    "make_tuple",
    "map_",
    "named_range",
    "neighbors",
    "reduce",
    "scan",
    "tuple_get",
    "unstructured_domain",
    *ARITHMETIC_BUILTINS,
    *TYPE_BUILTINS,
}

__all__ = [*BUILTINS]
