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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

import eve
from eve import NodeTranslator, NodeVisitor
from eve.utils import XIterator, xiter
from gtc import oir


@dataclass(frozen=True)
class Access:
    field: str
    offset: Tuple[int, int, int]
    is_write: bool

    @property
    def is_read(self) -> bool:
        return not self.is_write


class AccessCollector(NodeVisitor):
    """Collects all field accesses and corresponding offsets."""

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        accesses: List[Access],
        is_write: bool,
        **kwargs: Any,
    ) -> None:
        accesses.append(
            Access(
                field=node.name,
                offset=(node.offset.i, node.offset.j, node.offset.k),
                is_write=is_write,
            )
        )

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        **kwargs: Any,
    ) -> None:
        self.visit(node.right, is_write=False, **kwargs)
        self.visit(node.left, is_write=True, **kwargs)

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        **kwargs: Any,
    ) -> None:
        if node.mask:
            self.visit(node.mask, is_write=False, **kwargs)
        for stmt in node.body:
            self.visit(stmt, **kwargs)

    @dataclass
    class Result:
        _ordered_accesses: List["Access"]

        @staticmethod
        def _offset_dict(accesses: XIterator) -> Dict[str, Set[Tuple[int, int, int]]]:
            return accesses.reduceby(
                lambda acc, x: acc | {x.offset}, "field", init=set(), as_dict=True
            )

        def offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            return self._offset_dict(xiter(self._ordered_accesses))

        def read_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

        def write_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

        def fields(self) -> Set[str]:
            return {acc.field for acc in self._ordered_accesses}

        def read_fields(self) -> Set[str]:
            return {acc.field for acc in self._ordered_accesses if acc.is_read}

        def write_fields(self) -> Set[str]:
            return {acc.field for acc in self._ordered_accesses if acc.is_write}

        def ordered_accesses(self) -> List[Access]:
            return self._ordered_accesses

    @classmethod
    def apply(cls, node: oir.LocNode) -> "Result":
        result = cls.Result([])
        cls().visit(node, accesses=result._ordered_accesses)
        return result


class SymbolMapper(NodeTranslator):
    def visit_Node(
        self, node: eve.Node, *, symbol_mapper: Callable[[str], Optional[str]]
    ) -> eve.Node:
        translated = dict()
        for name, metadata in node.__node_children__.items():
            if (
                isinstance(metadata["definition"].type_, type)
                and issubclass(metadata["definition"].type_, (eve.SymbolName, eve.SymbolRef))
                and (mapped_name := symbol_mapper(getattr(node, name)))
            ):
                translated[name] = mapped_name
            else:
                translated[name] = self.visit(getattr(node, name), symbol_mapper=symbol_mapper)
        return node.__class__(**translated)

    @classmethod
    def apply(
        cls, node: eve.Node, symbol_mapper: Union[Callable[[str], Optional[str]], Mapping[str, str]]
    ) -> eve.Node:
        return cls().visit(
            node,
            symbol_mapper=lambda s: symbol_mapper.get(s, None)
            if isinstance(symbol_mapper, Mapping)
            else symbol_mapper,
        )
