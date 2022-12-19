# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
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

"""Definitions of basic Eve concepts."""


from __future__ import annotations

import ast
import copy
import re

from . import datamodels, exceptions
from . import extended_typing as xtyping
from . import trees, utils
from .datamodels import validators as _validators
from .extended_typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from .type_definitions import ConstrainedStr, IntEnum, StrEnum


_SYMBOL_NAME_RE: Final = re.compile(r"^[a-zA-Z_]\w*$")


class SymbolName(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """String value containing a valid symbol name for typical programming conventions."""

    __slots__ = ()


class SymbolRef(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """Reference to a symbol name."""

    __slots__ = ()


@datamodels.datamodel(slots=True, frozen=True)
class SourceLocation:
    """Source code location (line, column, source)."""

    line: int = datamodels.field(validator=_validators.ge(1))
    column: int = datamodels.field(validator=_validators.ge(1))
    source: str
    end_line: Optional[int] = datamodels.field(validator=_validators.optional(_validators.ge(1)))
    end_column: Optional[int] = datamodels.field(validator=_validators.optional(_validators.ge(1)))

    @classmethod
    def from_AST(cls, ast_node: ast.AST, source: Optional[str] = None) -> SourceLocation:
        if (
            not isinstance(ast_node, ast.AST)
            or getattr(ast_node, "lineno", None) is None
            or getattr(ast_node, "col_offset", None) is None
        ):
            raise ValueError(
                f"Passed AST node '{ast_node}' does not contain a valid source location."
            )
        if source is None:
            source = f"<ast.{type(ast_node).__name__} at 0x{id(ast_node):x}>"
        return cls(
            ast_node.lineno,
            ast_node.col_offset + 1,
            source,
            end_line=ast_node.end_lineno,
            end_column=ast_node.end_col_offset + 1 if ast_node.end_col_offset is not None else None,
        )

    def __init__(
        self,
        line: int,
        column: int,
        source: str,
        *,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> None:
        assert end_column is None or end_line is not None
        self.__auto_init__(  # type: ignore[attr-defined]  # __auto_init__ added dynamically
            line=line, column=column, source=source, end_line=end_line, end_column=end_column
        )

    def __str__(self) -> str:
        src = self.source or ""

        end_part = ""
        if self.end_line is not None:
            end_part += f" to {self.end_line}"
        if self.end_column is not None:
            end_part += f":{self.end_column}"

        return f"<{src}:{self.line}:{self.column}{end_part}>"


@datamodels.datamodel(slots=True, frozen=True)
class SourceLocationGroup:
    """A group of merged source code locations (with optional info)."""

    locations: Tuple[SourceLocation, ...] = datamodels.field(validator=_validators.non_empty())
    context: Optional[Union[str, Tuple[str, ...]]]

    def __init__(
        self, *locations: SourceLocation, context: Optional[Union[str, Tuple[str, ...]]] = None
    ) -> None:
        self.__auto_init__(locations=locations, context=context)  # type: ignore[attr-defined]  # __auto_init__ added dynamically

    def __str__(self) -> str:
        locs = ", ".join(str(loc) for loc in self.locations)
        context = f"#{self.context}#" if self.context else ""
        return f"<{context}[{locs}]>"


AnySourceLocation = Union[SourceLocation, SourceLocationGroup]

_T = TypeVar("_T")


class AnnexManager:
    register: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def register_user(
        cls: Type[AnnexManager],
        key: str,
        type_hint: xtyping.TypeAnnotation,
        *,
        shared: bool = False,
    ) -> Callable[[_T], _T]:
        assert isinstance(key, str)

        def _decorator(owner: _T) -> _T:
            if key in cls.register:
                reg_shared, reg_type, reg_owner = cls.register[key]
                if not shared:
                    raise exceptions.EveRuntimeError(
                        f"Annex key '{key}' has been already registered by {reg_owner}."
                    )
                if not reg_shared:
                    raise exceptions.EveRuntimeError(
                        f"Annex key '{key}' has been privately registered by {reg_owner}."
                    )
                elif type_hint != reg_type:
                    raise exceptions.EveRuntimeError(
                        f"Annex key '{key}' type '{type_hint}' does not match registered type '{reg_type}' "
                        f"registered by {reg_owner}."
                    )
                owners = reg_owner
            else:
                owners = []

            cls.register[key] = (shared, type_hint, [*owners, owner])

            return owner

        return _decorator


register_annex_user = AnnexManager.register_user


class Node(datamodels.DataModel, trees.Tree, kw_only=True):  # type: ignore[call-arg]  # kw_only from DataModel
    """Base class representing a node in a syntax tree.

    Implemented as a :class:`eve.datamodels.DataModel` with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`eve.datamodels.DataModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    The `annex` attribute is used to dynamically add data to a node, even to
    frozen classes. Data in the `annex` do not affect the hash or equality
    comparisons of the node, since it is not really a field. Thus, visitors
    and pipeline passes can freely attach computed attributes into the instance
    `annex`.
    """

    __slots__ = ()

    @property
    def annex(self) -> utils.Namespace:
        if not hasattr(self, "__node_annex__"):
            object.__setattr__(self, "__node_annex__", utils.Namespace())
        return self.__node_annex__  # type: ignore[attr-defined]  # __node_annex__ added dynamically

    def iter_children_values(self) -> Iterable:
        for name in self.__datamodel_fields__.keys():
            yield getattr(self, name)

    def iter_children_items(self) -> Iterable[Tuple[trees.TreeKey, Any]]:
        for name in self.__datamodel_fields__.keys():
            yield name, getattr(self, name)

    pre_walk_items = trees.pre_walk_items
    pre_walk_values = trees.pre_walk_values

    post_walk_items = trees.post_walk_items
    post_walk_values = trees.post_walk_values

    bfs_walk_items = trees.bfs_walk_items
    bfs_walk_values = trees.bfs_walk_values

    walk_items = trees.walk_items
    walk_values = trees.walk_values

    def copy(self: _T, update: Dict[str, Any]) -> _T:
        new_node = copy.deepcopy(self)
        for k, v in update.items():
            setattr(new_node, k, v)
        return new_node

    # TODO(egparedes): add useful hashes to base node
    # # @property
    # def content_id(self) -> int:
    #     ...
    #
    # @property
    # def annex_content_id(self) -> int:
    #     ...
    #
    # @property
    # def node_content_id(self) -> int:
    #     ...
    #
    # @property
    # def instance_content_id(self) -> int:
    #     ...
    #
    # @property
    # def instance_id(self) -> int:
    #     ...


NodeT = TypeVar("NodeT", bound="Node")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[NodeT, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
RootNode = Union[NodeT, CollectionNode]


class FrozenNode(Node, frozen=True):  # type: ignore[call-arg]  # frozen from DataModel
    ...


class GenericNode(datamodels.GenericDataModel, Node, kw_only=True):  # type: ignore[call-arg]  # kw_only from DataModel
    pass


class VType(datamodels.FrozenModel):

    # Unique name
    name: str


def eq_nonlocated(a: Node, b: Node) -> bool:
    """Compare two nodes, ignoring their `SourceLocation` or `SourceLocationGroup`."""
    return len(utils.ddiff(a, b, exclude_types=[SourceLocation, SourceLocationGroup])) == 0
