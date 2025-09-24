# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions of basic Eve concepts."""

from __future__ import annotations

import copy
import re

from . import datamodels, exceptions, extended_typing as xtyping, trees, utils
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
    """File-line-column information for source code."""

    filename: Optional[str]
    line: int = datamodels.field(validator=_validators.ge(1))
    column: int = datamodels.field(validator=_validators.ge(1))
    end_line: Optional[int] = datamodels.field(validator=_validators.optional(_validators.ge(1)))
    end_column: Optional[int] = datamodels.field(validator=_validators.optional(_validators.ge(1)))

    def __init__(
        self,
        filename: Optional[str],
        line: int,
        column: int,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> None:
        assert end_column is None or end_line is not None
        self.__auto_init__(  # type: ignore[attr-defined]  # __auto_init__ added dynamically
            filename=filename, line=line, column=column, end_line=end_line, end_column=end_column
        )

    def __str__(self) -> str:
        filename_str = self.filename or "-"

        end_line_str = self.end_line if self.end_line is not None else "-"
        end_column_str = self.end_column if self.end_column is not None else "-"

        end_str: Optional[str] = None
        if self.end_column is not None:
            end_str = f"{end_line_str}:{end_column_str}"
        elif self.end_line is not None:
            end_str = f"{end_line_str}"

        if end_str is not None:
            return f"<{filename_str}:{self.line}:{self.column} to {end_str}>"
        return f"<{filename_str}:{self.line}:{self.column}>"


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
    `annex`. Note that `annex` attribute is not implicitly copied in the
    :class:`NodeTranslator`. If you want it to persist across multiple
    :class:`NodeTranslator`s either use a `root_validator` to dynamically
    (re)compute the annex on node construction (see e.g.
    :class:`SymbolTableTrait`) or add the parts of the annex that should be
    preserved to the `PRESERVED_ANNEX_ATTRS` attribute in the
    :class:`NodeTranslator` class..
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


NodeT = TypeVar("NodeT", bound="Node")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[NodeT, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
RootNode = Union[NodeT, CollectionNode]


class FrozenNode(Node, frozen=True):  # type: ignore[call-arg]  # frozen from DataModel
    ...


class GenericNode(datamodels.GenericDataModel, Node, kw_only=True):  # type: ignore[call-arg]  # kw_only from DataModel
    pass


def eq_nonlocated(a: Node, b: Node) -> bool:
    """Compare two nodes, ignoring their `SourceLocation` or `SourceLocationGroup`."""
    return len(utils.ddiff(a, b, exclude_types=[SourceLocation, SourceLocationGroup])) == 0
