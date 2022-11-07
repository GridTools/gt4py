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

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, cast

import eve
import eve.utils
from gtc import common, oir
from gtc.definitions import Extent
from gtc.passes.horizontal_masks import mask_overlap_with_extent


OffsetT = TypeVar("OffsetT")

GeneralOffsetTuple = Tuple[int, int, Optional[int]]

_digits_at_end_pattern = re.compile(r"[0-9]+$")
_generated_name_pattern = re.compile(r".+_gen_[0-9]+")


@dataclass(frozen=True)
class GenericAccess(Generic[OffsetT]):
    field: str
    offset: OffsetT
    is_write: bool
    data_index: List[oir.Expr] = dataclasses.field(default_factory=list)
    horizontal_mask: Optional[common.HorizontalMask] = None

    @property
    def is_read(self) -> bool:
        return not self.is_write

    def to_extent(self, horizontal_extent: Extent) -> Optional[Extent]:
        """
        Convert the access to an extent provided a horizontal extent for the access.

        This returns None if no overlap exists between the horizontal mask and interval.
        """
        offset_as_extent = Extent.from_offset(cast(Tuple[int, int, int], self.offset)[:2])
        zeros = Extent.zeros(ndims=2)
        if self.horizontal_mask:
            if dist_from_edge := mask_overlap_with_extent(self.horizontal_mask, horizontal_extent):
                return ((horizontal_extent - dist_from_edge) + offset_as_extent) | zeros
            else:
                return None
        else:
            return horizontal_extent + offset_as_extent


class CartesianAccess(GenericAccess[Tuple[int, int, int]]):
    pass


class GeneralAccess(GenericAccess[GeneralOffsetTuple]):
    pass


AccessT = TypeVar("AccessT", bound=GenericAccess)


class AccessCollector(eve.NodeVisitor):
    """Collects all field accesses and corresponding offsets."""

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        accesses: List[GeneralAccess],
        is_write: bool,
        horizontal_mask: Optional[common.HorizontalMask] = None,
        **kwargs: Any,
    ) -> None:
        self.generic_visit(node, accesses=accesses, is_write=is_write, **kwargs)
        offsets = node.offset.to_dict()
        accesses.append(
            GeneralAccess(
                field=node.name,
                offset=(cast(int, offsets["i"]), cast(int, offsets["j"]), offsets["k"]),
                data_index=node.data_index,
                is_write=is_write,
                horizontal_mask=horizontal_mask,
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
        self.visit(node.body, **kwargs)

    def visit_While(self, node: oir.While, **kwargs: Any) -> None:
        self.visit(node.cond, is_write=False, **kwargs)
        self.visit(node.body, **kwargs)

    def visit_HorizontalRestriction(self, node: oir.HorizontalRestriction, **kwargs: Any) -> None:
        self.visit(node.body, horizontal_mask=node.mask, **kwargs)

    @dataclass
    class GenericAccessCollection(Generic[AccessT, OffsetT]):
        _ordered_accesses: List[AccessT]

        @staticmethod
        def _offset_dict(accesses: eve.utils.XIterable) -> Dict[str, Set[OffsetT]]:
            return accesses.reduceby(
                lambda acc, x: acc | {x.offset}, "field", init=set(), as_dict=True
            )

        def offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictionary, mapping all accessed fields' names to sets of offset tuples."""
            return self._offset_dict(eve.utils.XIterable(self._ordered_accesses))

        def read_offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictionary, mapping read fields' names to sets of offset tuples."""
            return self._offset_dict(
                eve.utils.XIterable(self._ordered_accesses).filter(lambda x: x.is_read)
            )

        def read_accesses(self) -> List[AccessT]:
            """Get the sub-list of read accesses."""
            return list(eve.utils.XIterable(self._ordered_accesses).filter(lambda x: x.is_read))

        def write_offsets(self) -> Dict[str, Set[OffsetT]]:
            """Get a dictionary, mapping written fields' names to sets of offset tuples."""
            return self._offset_dict(
                eve.utils.XIterable(self._ordered_accesses).filter(lambda x: x.is_write)
            )

        def write_accesses(self) -> List[AccessT]:
            """Get the sub-list of write accesses."""
            return list(eve.utils.XIterable(self._ordered_accesses).filter(lambda x: x.is_write))

        def fields(self) -> Set[str]:
            """Get a set of all accessed fields' names."""
            return {acc.field for acc in self._ordered_accesses}

        def read_fields(self) -> Set[str]:
            """Get a set of all read fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_read}

        def write_fields(self) -> Set[str]:
            """Get a set of all written fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_write}

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
                        offset=cast(Tuple[int, int, int], acc.offset),
                        data_index=acc.data_index,
                        is_write=acc.is_write,
                    )
                    for acc in self._ordered_accesses
                    if acc.offset[2] is not None
                ]
            )

        def has_variable_access(self) -> bool:
            return any(acc.offset[2] is None for acc in self._ordered_accesses)

    @classmethod
    def apply(cls, node: eve.RootNode, **kwargs: Any) -> "AccessCollector.GeneralAccessCollection":
        result = cls.GeneralAccessCollection([])
        cls().visit(node, accesses=result._ordered_accesses, **kwargs)
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


def collect_symbol_names(node: eve.RootNode) -> Set[str]:
    return (
        eve.walk_values(node)
        .if_isinstance(eve.SymbolTableTrait)
        .getattr("annex")
        .getattr("symtable")
        .reduce(lambda names, symtable: names.union(symtable.keys()), init=set())
    )


class StencilExtentComputer(eve.NodeVisitor):
    @dataclass
    class Context:
        # TODO: Remove dependency on gt4py.definitions here
        fields: Dict[str, Extent] = dataclasses.field(default_factory=dict)
        blocks: Dict[int, Extent] = dataclasses.field(default_factory=dict)

    def __init__(self, add_k: bool = False):
        self.add_k = add_k
        self.zero_extent = Extent.zeros(ndims=2)

    def visit_Stencil(self, node: oir.Stencil) -> "Context":
        ctx = self.Context()
        for vloop in reversed(node.vertical_loops):
            self.visit(vloop, ctx=ctx)

        if self.add_k:
            ctx.fields = {name: Extent(*extent, (0, 0)) for name, extent in ctx.fields.items()}

        for name in (p.name for p in node.params if p.name not in ctx.fields):
            ctx.fields[name] = self.zero_extent

        return ctx

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs: Any) -> None:
        for hexec in reversed(node.horizontal_executions):
            self.visit(hexec, **kwargs)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, ctx: Context) -> None:
        results = AccessCollector.apply(node)

        horizontal_extent = self.zero_extent
        for access in (acc for acc in results.ordered_accesses() if acc.is_write):
            horizontal_extent |= ctx.fields.setdefault(access.field, self.zero_extent)
        ctx.blocks[id(node)] = horizontal_extent

        for access in results.ordered_accesses():
            extent = access.to_extent(horizontal_extent)
            if extent is None:
                continue

            if access.field in ctx.fields:
                ctx.fields[access.field] = ctx.fields[access.field] | extent
            else:
                ctx.fields[access.field] = extent


def compute_horizontal_block_extents(node: oir.Stencil, **kwargs: Any) -> Dict[int, Extent]:
    ctx = StencilExtentComputer(**kwargs).visit(node)
    return ctx.blocks


def compute_fields_extents(node: oir.Stencil, **kwargs: Any) -> Dict[str, Extent]:
    ctx = StencilExtentComputer(**kwargs).visit(node)
    return ctx.fields


def compute_extents(
    node: oir.Stencil, **kwargs: Any
) -> Tuple[Dict[str, Extent], Dict[int, Extent]]:
    ctx = StencilExtentComputer(**kwargs).visit(node)
    return ctx.fields, ctx.blocks
