# GT4Py - GridTools Framework
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

import boltons.typeutils  # type: ignore[import]

from gt4py.eve import NodeVisitor
from gt4py.next.iterator import ir


ALL_NEIGHBORS = boltons.typeutils.make_sentinel("ALL_NEIGHBORS")


class CollectShifts(NodeVisitor):
    """Collects shifts applied to symbol references.

    Fills the provided `shifts` keyword argument (of type `dict[str, list[tuple]]`)
    with a list of offset tuples. E.g., if there is just `deref(x)` and a
    `deref(shift(a, b)(x))` in the node tree, the result will be
    `{"x": [(), (a, b)]}`.

    For reductions, the special value `ALL_NEIGHBORS` is used. E.g,
    `reduce(f, 0.0)(shift(V2E)(x))` will return `{"x": [(V2E, ALL_NEIGHBORS)]}`.

    Limitations:
    - Nested shift calls like `deref(shift(c, d)(shift(a, b)(x)))` are not supported.
      That is, all shifts must be normalized (that is, `deref(shift(a, b, c, d)(x))`
      works in the given example).
    - Calls to lift and scan are not supported.
    """

    @staticmethod
    def _as_deref(node: ir.FunCall):
        if node.fun == ir.SymRef(id="deref"):
            (arg,) = node.args
            if isinstance(arg, ir.SymRef):
                return arg.id

    @staticmethod
    def _as_shift(node: ir.Expr):
        if isinstance(node, ir.FunCall) and node.fun == ir.SymRef(id="shift"):
            return tuple(node.args)

    @classmethod
    def _as_shift_call(cls, node: ir.Expr):
        if (
            isinstance(node, ir.FunCall)
            and (offsets := cls._as_shift(node.fun))
            and isinstance(sym := node.args[0], ir.SymRef)
        ):
            return sym.id, offsets

    @classmethod
    def _as_deref_shift(cls, node: ir.FunCall):
        if node.fun == ir.SymRef(id="deref"):
            (arg,) = node.args
            if sym_and_offsets := cls._as_shift_call(arg):
                return sym_and_offsets

    @staticmethod
    def _as_reduce(node: ir.FunCall):
        if isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="reduce"):
            assert len(node.fun.args) == 2
            return node.args

    def visit_FunCall(self, node: ir.FunCall, *, shifts: dict[str, list[tuple]]):
        if sym_id := self._as_deref(node):
            # direct deref of a symbol: deref(sym)
            shifts.setdefault(sym_id, []).append(())
            return
        if sym_and_offsets := self._as_deref_shift(node):
            # deref of a shifted symbol: deref(shift(...)(sym))
            sym, offsets = sym_and_offsets
            shifts.setdefault(sym, []).append(offsets)
            return
        if sym_and_offsets := self._as_shift_call(node):
            # just shifting: shift(...)(sym)
            # required to catch ‘underefed’ shifts in reduction calls
            sym, offsets = sym_and_offsets
            shifts.setdefault(sym, []).append(offsets)
            return
        if reduction_args := self._as_reduce(node):
            # reduce(..., ...)(args...)
            nested_shifts = dict[str, list[tuple]]()
            self.visit(reduction_args, shifts=nested_shifts)
            for sym, offset_list in nested_shifts.items():
                for offsets in offset_list:
                    shifts.setdefault(sym, []).append(offsets + (ALL_NEIGHBORS,))
            return

        if not isinstance(node.fun, ir.SymRef) or node.fun.id in ("lift", "scan"):
            raise ValueError(f"Unsupported node: {node}")

        self.generic_visit(node, shifts=shifts)
