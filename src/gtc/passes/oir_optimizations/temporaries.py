# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

from eve import NodeMutator, NodeTranslator, SymbolTableTrait
from gtc import oir

from .utils import AccessCollector, collect_symbol_names, symbol_name_creator


class TemporariesToScalarsBase(NodeTranslator):
    contexts = (SymbolTableTrait.symtable_merger,)

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, tmps_name_map: Dict[str, str], **kwargs: Any
    ) -> Union[oir.FieldAccess, oir.ScalarAccess]:
        if node.name in tmps_name_map:
            assert (
                node.offset.i == node.offset.j == node.offset.k == 0
            ), "Non-zero offset in temporary that is replaced?!"
            return oir.ScalarAccess(name=tmps_name_map[node.name], dtype=node.dtype)
        return self.generic_visit(node, tmps_name_map=tmps_name_map, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        tmps_to_replace: Set[str],
        symtable: Dict[str, Any],
        new_symbol_name: Callable[[str], str],
        **kwargs: Any,
    ) -> oir.HorizontalExecution:
        local_tmps_to_replace = (
            node.iter_tree()
            .if_isinstance(oir.FieldAccess)
            .getattr("name")
            .if_in(tmps_to_replace)
            .to_set()
        )
        tmps_name_map = {tmp: new_symbol_name(tmp) for tmp in local_tmps_to_replace}

        return oir.HorizontalExecution(
            body=self.visit(node.body, tmps_name_map=tmps_name_map, symtable=symtable, **kwargs),
            declarations=node.declarations
            + [
                oir.LocalScalar(
                    name=tmps_name_map[tmp], dtype=symtable[tmp].dtype, loc=symtable[tmp].loc
                )
                for tmp in local_tmps_to_replace
            ],
            loc=node.loc,
        )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        tmps_to_replace: Set[str],
        **kwargs: Any,
    ) -> oir.VerticalLoop:
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=self.visit(node.sections, tmps_to_replace=tmps_to_replace, **kwargs),
            caches=[c for c in node.caches if c.name not in tmps_to_replace],
            loc=node.loc,
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        tmps_to_replace = kwargs["tmps_to_replace"]
        all_names = collect_symbol_names(node)
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(
                node.vertical_loops,
                new_symbol_name=symbol_name_creator(all_names),
                **kwargs,
            ),
            declarations=[d for d in node.declarations if d.name not in tmps_to_replace],
            loc=node.loc,
        )


class LocalTemporariesToScalars(TemporariesToScalarsBase):
    """Replaces temporary fields accessed only within a single horizontal execution by scalars.

    1. Finds temporaries that are only accessed within a single HorizontalExecution.
    2. Replaces corresponding FieldAccess nodes by ScalarAccess nodes.
    3. Removes matching temporaries from VerticalLoop declarations.
    4. Add matching temporaries to HorizontalExecution declarations.
    """

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        horizontal_executions = node.iter_tree().if_isinstance(oir.HorizontalExecution)
        counts: collections.Counter = sum(
            (
                collections.Counter(
                    horizontal_execution.iter_tree()
                    .if_isinstance(oir.FieldAccess)
                    .getattr("name")
                    .if_in({tmp.name for tmp in node.declarations})
                    .to_set()
                )
                for horizontal_execution in horizontal_executions
            ),
            collections.Counter(),
        )
        local_tmps = {tmp for tmp, count in counts.items() if count == 1}
        return super().visit_Stencil(node, tmps_to_replace=local_tmps, **kwargs)


class WriteBeforeReadTemporariesToScalars(TemporariesToScalarsBase):
    """Replaces temporay fields that are always written before read by scalars."""

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        write_before_read_tmps = {
            symbol
            for symbol, value in kwargs["symtable"].items()
            if isinstance(value, oir.Temporary)
        }
        horizontal_executions = node.iter_tree().if_isinstance(oir.HorizontalExecution)

        for horizontal_execution in horizontal_executions:
            accesses = AccessCollector.apply(horizontal_execution)
            offsets = accesses.offsets()
            ordered_accesses = accesses.ordered_accesses()

            def write_before_read(tmp: str) -> bool:
                if tmp not in offsets:
                    return True
                if offsets[tmp] != {(0, 0, 0)}:
                    return False
                return next(o.is_write for o in ordered_accesses if o.field == tmp)

            write_before_read_tmps = {
                tmp for tmp in write_before_read_tmps if write_before_read(tmp)
            }

        return super().visit_Stencil(node, tmps_to_replace=write_before_read_tmps, **kwargs)


def _iter_stmt_pairs(stencil: oir.Stencil, reverse: bool = False):
    def _collect_stmts_pairs_rev(stmts: Sequence[oir.Stmt]):
        pairs = []
        iterator = reversed(stmts) if reverse else stmts
        for stmt in iterator:
            if isinstance(stmt, oir.AssignStmt):
                pairs.append((stmt.left, stmt.right))
            elif isinstance(stmt, oir.MaskStmt):
                pairs.append((None, stmt.mask))
                pairs.extend(_collect_stmts_pairs_rev(stmt.body))
            else:
                raise TypeError("Unrecognized oir.Stmt subtype")
        return pairs

    for loop in reversed(stencil.vertical_loops) if reverse else stencil.vertical_loops:
        for section in loop.sections:
            for hexec in (
                reversed(section.horizontal_executions)
                if reverse
                else section.horizontal_executions
            ):
                yield from _collect_stmts_pairs_rev(hexec.body)


def _annotate_reads(stencil: oir.Stencil) -> Dict[int, Set[str]]:
    """
    For each statement, determines the set of temporary fields that are read at a later stmt.

    Returns
    -------
        Dict[int, Set[str]]
            Map from stmt id to set of temporary fields read at a later stmt.
    """
    temporaries = {d.name for d in stencil.declarations}

    reads_after: Dict[int, Set[str]] = {}
    last_reads: Set[str] = set()
    for _, rhs in _iter_stmt_pairs(stencil, reverse=True):
        current_reads: Set[str] = (
            rhs.iter_tree()
            .if_isinstance(oir.FieldAccess)
            .filter(lambda acc: acc.name in temporaries)
            .getattr("name")
            .to_set()
        )
        reads_after[id(rhs)] = last_reads.copy()
        last_reads |= current_reads

    return reads_after


class _NameRemapper(NodeMutator):
    def visit_FieldAccess(self, node: oir.FieldAccess, *, symbol_to_temp: Dict[str, str]):
        if node.name in symbol_to_temp:
            node.name = symbol_to_temp[node.name]
        return node


def _find_temporary(
    declarations: List[oir.Temporary], unused_allocated: Set[str], symbol: str
) -> Optional[str]:
    """Find a suitable temporary for _remap_temporaries."""
    lval_decl = next(decl for decl in declarations if decl.name == symbol)
    for decl in (d for d in declarations if d.name in unused_allocated):
        if (
            lval_decl.dtype == decl.dtype
            and lval_decl.dimensions == decl.dimensions
            and lval_decl.data_dims == decl.data_dims
        ):
            return decl.name

    return None


def _remap_temporaries(
    stencil: oir.Stencil, symbol_reads_after: Dict[int, Set[str]]
) -> oir.Stencil:
    all_symbols = {d.name for d in stencil.declarations}
    in_use_allocated: Set[str] = set()
    unused_allocated: Set[str] = set()
    symbol_to_temp: Dict[str, str] = {}

    for lhs, rhs in _iter_stmt_pairs(stencil, reverse=False):
        lval_symbol = lhs.name if lhs is not None else None
        if lval_symbol is not None and lval_symbol in all_symbols:
            if lval_symbol in symbol_to_temp:
                lhs.name = symbol_to_temp[lval_symbol]
            elif lval_temp := _find_temporary(stencil.declarations, unused_allocated, lval_symbol):
                unused_allocated.remove(lval_temp)
                in_use_allocated.add(lval_temp)
                symbol_to_temp[lval_symbol] = lval_temp
                lhs.name = lval_temp
            else:
                in_use_allocated.add(lval_symbol)
                symbol_to_temp[lval_symbol] = lval_symbol
                # lhs.name is up to date
        else:
            pass

        temps_read_after = {
            symbol_to_temp[name] for name in symbol_reads_after[id(rhs)] if name in symbol_to_temp
        }
        unused_allocated = unused_allocated | {
            name for name in in_use_allocated if name not in temps_read_after
        }
        in_use_allocated = {name for name in in_use_allocated if name in temps_read_after}

        _NameRemapper().visit(rhs, symbol_to_temp=symbol_to_temp)

    stencil.declarations = [
        decl for decl in stencil.declarations if decl.name in set(symbol_to_temp.values())
    ]

    return stencil


def fold_temporary_fields(node: oir.Stencil) -> oir.Stencil:
    """
    Re-use oir.Temporary declarations where possible, without reordering statements, to reduce memory pressure.

    Postcondition: The same or fewer temporaries referenced and declarations in stencil.declarations.

    Parameters
    ----------
    node `oir.Stencil`
        The Stencil to be transformed.

    Returns
    -------
    oir.Stencil
        New copy of the stencil, with temporaries folded.

    """
    new_stencil = copy.deepcopy(node)
    reads_after = _annotate_reads(new_stencil)
    return _remap_temporaries(new_stencil, reads_after)
