# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import functools
from typing import Any, Callable, Dict, Iterator, List, TypeVar

import numpy as np

import eve
from eve import datamodels
from gtc import gtir
from gtc.common import DataType, op_to_ufunc, typestr_to_data_type
from gtc.gtir import Expr


def _upcast_node(target_dtype: DataType, node: Expr) -> Expr:
    return node if node.dtype == target_dtype else gtir.Cast(dtype=target_dtype, expr=node)


def _upcast_nodes(*exprs: Expr, upcasting_rule: Callable) -> Iterator[Expr]:
    assert all(e.dtype for e in exprs)
    dtypes: List[DataType] = [e.dtype for e in exprs]  # type: ignore # guaranteed to be not None
    target_dtypes = upcasting_rule(*dtypes)
    return iter(_upcast_node(target_dtype, arg) for target_dtype, arg in zip(target_dtypes, exprs))


_T = TypeVar("_T", bound=eve.Node)


def _update_node(node: _T, updated_children: Dict[str, eve.RootNode]) -> _T:
    # create new node only if children changed
    old_children = datamodels.asdict(node)
    if any([old_children[k] != updated_children[k] for k in updated_children.keys()]):
        return node.copy(update=updated_children)
    else:
        return node


@functools.lru_cache
def _numpy_ufunc_upcasting_rule(*dtypes, ufunc: np.ufunc):
    """
    Look up upcasting behavior according to NumPy universal function casting convention.

    NumPy specifies that it chooses ufunc implementations based on input types, where the inputs are suitably cast if
    necessary. Mimicking this results in a behavior that can be reproduced in C++ backends but is also consistent with
    python in the numpy backend to the extent possible. We choose ufunc implementations according to NumPy rules while
    implementing C/C++ type promotion rules, restricted to supported types.
    See https://numpy.org/doc/stable/user/basics.ufuncs.html?highlight=index#type-casting-rules for details.
    """
    for t in ufunc.types:
        inputs, output = t.split("->")
        assert len(inputs) == len(dtypes)
        if all(
            arg_dtype <= typestr_to_data_type(np.dtype(cand_typestr).str)
            for arg_dtype, cand_typestr in zip(dtypes, inputs)
        ):
            assert typestr_to_data_type(np.dtype(output).str) != DataType.INVALID
            return [typestr_to_data_type(np.dtype(cand_typestr).str) for cand_typestr in inputs]
    raise ValueError(f"No implementation found for dtypes {dtypes} and ufunc {ufunc}")


def _common_upcasting_rule(*dtypes):
    """Look up upcasting behavior according to C++ casting rules."""
    if all(dtype == DataType.DEFAULT for dtype in dtypes):
        res_type = DataType.DEFAULT
    else:
        res_type = max(dtypes)
    return [res_type] * len(dtypes)


class _GTIRUpcasting(eve.NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes.

    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs: Any) -> gtir.BinaryOp:
        upcasting_rule = functools.partial(_numpy_ufunc_upcasting_rule, ufunc=op_to_ufunc(node.op))
        left, right = _upcast_nodes(
            self.visit(node.left), self.visit(node.right), upcasting_rule=upcasting_rule
        )
        return _update_node(node, {"left": left, "right": right})

    def visit_UnaryOp(self, node: gtir.UnaryOp, **kwargs: Any) -> gtir.UnaryOp:
        upcasting_rule = functools.partial(_numpy_ufunc_upcasting_rule, ufunc=op_to_ufunc(node.op))
        (expr,) = _upcast_nodes(self.visit(node.expr), upcasting_rule=upcasting_rule)
        return _update_node(node, {"expr": expr})

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs: Any) -> gtir.TernaryOp:
        true_expr, false_expr = _upcast_nodes(
            self.visit(node.true_expr),
            self.visit(node.false_expr),
            upcasting_rule=_common_upcasting_rule,
        )
        return _update_node(
            node, {"true_expr": true_expr, "false_expr": false_expr, "cond": self.visit(node.cond)}
        )

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs: Any) -> gtir.NativeFuncCall:
        upcasting_rule = functools.partial(
            _numpy_ufunc_upcasting_rule, ufunc=op_to_ufunc(node.func)
        )
        args = [*_upcast_nodes(*self.visit(node.args), upcasting_rule=upcasting_rule)]
        return _update_node(node, {"args": args})

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs: Any) -> gtir.ParAssignStmt:
        assert node.left.dtype

        def upcasting_rule(*dtypes):
            return [node.left.dtype] * len(dtypes)

        _, right = _upcast_nodes(node.left, self.visit(node.right), upcasting_rule=upcasting_rule)
        return _update_node(node, {"right": right})


def upcast(node: gtir.Stencil) -> gtir.Stencil:
    return _GTIRUpcasting().visit(node)
