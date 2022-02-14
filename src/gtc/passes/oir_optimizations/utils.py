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

import functools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, cast

from eve import NodeVisitor
from eve.concepts import TreeNode
from eve.traits import SymbolTableTrait
from eve.utils import XIterable, xiter
from gt4py.definitions import Extent
from gtc import common, oir


OffsetT = TypeVar("OffsetT")

GeneralOffsetTuple = Tuple[int, int, Optional[int]]
HorizontalExtent = Tuple[Tuple[int, int], Tuple[int, int]]


_digits_at_end_pattern = re.compile(r"[0-9]+$")
_generated_name_pattern = re.compile(r".+_gen_[0-9]+")


@dataclass(frozen=True)
class GenericAccess(Generic[OffsetT]):
    field: str
    offset: OffsetT
    is_write: bool
    in_mask: bool = False
    region: Optional[oir.HorizontalMask] = None

    @property
    def is_read(self) -> bool:
        return not self.is_write


class CartesianAccess(GenericAccess[Tuple[int, int, int]]):
    pass


class GeneralAccess(GenericAccess[GeneralOffsetTuple]):
    pass


AccessT = TypeVar("AccessT", bound=GenericAccess)


class AccessCollector(NodeVisitor):
    """Collects all field accesses and corresponding offsets."""

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        accesses: List[GeneralAccess],
        is_write: bool,
        region: oir.HorizontalMask = None,
        in_mask=False,
        **kwargs: Any,
    ) -> None:
        self.visit(node.offset, accesses=accesses, is_write=False, region=region, in_mask=in_mask)
        accesses.append(
            GeneralAccess(
                field=node.name,
                offset=node.offset.to_tuple(),
                is_write=is_write,
                in_mask=in_mask,
                region=region,
            )
        )

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        **kwargs: Any,
    ) -> None:
        self.visit(node.right, is_write=False, **kwargs)
        self.visit(node.left, is_write=True, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> None:

        self.visit(node.mask, is_write=False, **kwargs)
        regions = node.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list()
        if regions:
            for region in regions:
                self.visit(node.body, in_mask=True, region=region, **kwargs)
        else:
            self.visit(node.body, in_mask=True, region=None, **kwargs)

    @dataclass
    class GenericAccessCollection(Generic[AccessT, OffsetT]):
        _ordered_accesses: List[AccessT]

        @staticmethod
        def _offset_dict(accesses: XIterable) -> Dict[str, Set[OffsetT]]:
            return accesses.reduceby(
                lambda acc, x: acc | {x.offset}, "field", init=set(), as_dict=True
            )

        def offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictonary, mapping all accessed fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses))

        def read_offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictonary, mapping read fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

        def read_accesses(self) -> List[AccessT]:
            """Get the sub-list of read accesses."""
            return list(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

        def write_offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictonary, mapping written fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

        def write_accesses(self) -> List[AccessT]:
            """Get the sub-list of write accesses."""
            return list(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

        def fields(self) -> Set[str]:
            """Get a set of all accessed fields' names."""
            return {acc.field for acc in self._ordered_accesses}

        def read_fields(self) -> Set[str]:
            """Get a set of all read fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_read}

        def write_fields(self) -> Set[str]:
            """Get a set of all written fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_write}

        def mask_writes(self) -> Set[str]:
            """Get a set of all fields' names written in mask statements."""
            return {acc.field for acc in self._ordered_accesses if acc.is_write and acc.in_mask}

        def ordered_accesses(self) -> List[AccessT]:
            """Get a list of ordered accesses."""
            return self._ordered_accesses

    class CartesianAccessCollection(GenericAccessCollection[CartesianAccess, Tuple[int, int, int]]):
        pass

    class GeneralAccessCollection(GenericAccessCollection[GeneralAccess, GeneralOffsetTuple]):
        def cartesian_accesses(self) -> "AccessCollector.CartesianAccessCollection":
            return AccessCollector.CartesianAccessCollection(
                [
                    CartesianAccess(
                        field=acc.field,
                        offset=cast(Tuple[int, int, int], acc.offset)
                        if acc.offset[2] is not None
                        else (0, 0, 0),
                        is_write=acc.is_write,
                        region=acc.region,
                        in_mask=acc.in_mask,
                    )
                    for acc in self._ordered_accesses
                ]
            )

        def has_variable_access(self) -> bool:
            return any(acc.offset[2] is None for acc in self._ordered_accesses)

    @classmethod
    def _compensate_regions(cls, result: "AccessCollector.GenericAccessCollection"):

        res_accesses = []
        for acc in result._ordered_accesses:
            if acc.region is not None:
                res_offset = []
                for off, bound in zip(acc.offset, (acc.region.i, acc.region.j)):
                    bound = common.HorizontalInterval(start=bound.start, end=bound.end)
                    if off > 0:
                        if bound.end is None:
                            pass
                        elif bound.end.level == common.LevelMarker.END:
                            off = off + bound.end.offset
                        else:
                            off = 0
                    if off < 0:
                        if bound.start is None:
                            pass
                        elif bound.start.level == common.LevelMarker.START:
                            off = off + bound.start.offset
                        else:
                            off = 0
                    res_offset.append(off)
                res_offset.append(acc.offset[2])
                res_accesses.append(
                    GeneralAccess(
                        field=acc.field,
                        offset=(res_offset[0], res_offset[1], res_offset[2]),
                        is_write=acc.is_write,
                        in_mask=acc.in_mask,
                        region=acc.region,
                    )
                )
            else:
                res_accesses.append(acc)
        return AccessCollector.GenericAccessCollection(res_accesses)

    @classmethod
    def apply(
        cls, node: TreeNode, *, compensate_regions=False, **kwargs: Any
    ) -> "AccessCollector.GeneralAccessCollection":
        result = cls.GeneralAccessCollection([])
        cls().visit(node, accesses=result._ordered_accesses, **kwargs)

        if compensate_regions:
            result = cls._compensate_regions(result)
            return result

        return result


def symbol_name_creator(used_names: Set[str]) -> Callable[[str], str]:
    """Create a function that generates symbol names that are not already in use.

    Args:
        used_names: Symbol names that are already in use and thus should not be generated.
                    NOTE: `used_names` will be modified to contain all generated symbols.

    Returns:
        A callable to generate new unique symbol names.
    """

    def increment_string_suffix(s: str) -> str:
        if not _generated_name_pattern.match(s):
            return s + "_gen_0"
        return _digits_at_end_pattern.sub(lambda n: str(int(n.group()) + 1), s)

    def new_symbol_name(name: str) -> str:
        while name in used_names:
            name = increment_string_suffix(name)
        used_names.add(name)
        return name

    return new_symbol_name


def collect_symbol_names(node: TreeNode) -> Set[str]:
    return (
        node.iter_tree()
        .if_isinstance(SymbolTableTrait)
        .getattr("symtable_")
        .reduce(lambda names, symtable: names.union(symtable.keys()), init=set())
    )


class _HorizontalExecutionExtents(NodeVisitor):
    @dataclass
    class Context:
        # TODO: Remove dependency on gt4py.definitions here
        field_extents: Dict[str, Extent] = field(default_factory=dict)
        block_extents: Dict[int, Extent] = field(default_factory=dict)

    def visit_Stencil(self, node: oir.Stencil) -> Dict[int, Extent]:
        ctx = self.Context()
        for vloop in reversed(node.vertical_loops):
            self.visit(vloop, ctx=ctx)

        return ctx.block_extents

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs: Any) -> None:
        for hexec in reversed(node.horizontal_executions):
            self.visit(hexec, **kwargs)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, ctx: Context) -> None:
        results = AccessCollector.apply(node).cartesian_accesses()
        horizontal_extent = functools.reduce(
            lambda ext, name: ext | ctx.field_extents.get(name, Extent.zeros(ndims=2)),
            results.write_fields(),
            Extent.zeros(ndims=2),
        )
        ctx.block_extents[id(node)] = horizontal_extent

        for name, accesses in results.read_offsets().items():
            extent = functools.reduce(
                lambda ext, off: ext | Extent.from_offset(off[:2]), accesses, Extent.zeros(ndims=2)
            )
            ctx.field_extents[name] = ctx.field_extents.get(name, Extent.zeros(ndims=2)).union(
                horizontal_extent + extent
            )


def compute_horizontal_block_extents(node: oir.Stencil) -> Dict[int, Extent]:
    return _HorizontalExecutionExtents().visit(node)
