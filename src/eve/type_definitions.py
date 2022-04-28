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

"""Definitions of useful field and general types."""


from __future__ import annotations

import ast
import enum
import functools
import re

import boltons.typeutils
import pydantic
import xxhash
from boltons.typeutils import classproperty  # noqa: F401
from pydantic import validator  # noqa
from pydantic import NegativeFloat, NegativeInt, PositiveFloat, PositiveInt  # noqa
from pydantic import StrictBool as Bool  # noqa: F401
from pydantic import StrictFloat as Float  # noqa: F401
from pydantic import StrictInt as Int  # noqa: F401
from pydantic import StrictStr as Str
from pydantic.types import ConstrainedStr

from .extended_typing import Any, Callable, Generator, Optional, Tuple, Type, Union


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = boltons.typeutils.make_sentinel(name="NOTHING", var_name="NOTHING")


#: Typing definitions for `__get_validators__()` methods
# (defined but not exported in `pydantic.typing`)
PydanticCallableGenerator = Generator[Callable[..., Any], None, None]


#: :class:`bytes subclass for strict field definition
Bytes = bytes


class Enum(enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> Enum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class IntEnum(enum.IntEnum):
    """Basic :class:`enum.IntEnum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> IntEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class StrEnum(str, enum.Enum):
    """:class:`enum.Enum` subclass with strict type validation and supporting string operations."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> StrEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v

    def __str__(self) -> str:
        assert isinstance(self.value, str)
        return self.value


class SymbolName(ConstrainedStr):
    """Name of a symbol.

    The name itself is only validated automatically within a Pydantic
    model validation context. Use :meth:`from_string` to create a properly
    validated isolated instance.

    """

    #: Regular expression used to validate the name string
    regex = re.compile(r"^[a-zA-Z_]\w*$")
    strict = True

    @classmethod
    def from_string(cls, name: str) -> SymbolName:
        """Self-validated instance creation."""
        name = cls.validate(name)
        return cls(name)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def constrained(pattern: Union[str, re.Pattern]) -> Type[SymbolName]:
        """Create a new SymbolName subclass using the provided string as validation RE."""
        if isinstance(pattern, re.Pattern):
            regex = pattern
            pattern = pattern.pattern
        else:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                raise TypeError(f"Invalid regular expression definition:  '{pattern}'.") from e

        assert isinstance(pattern, str)
        xxh64 = xxhash.xxh64()
        xxh64.update(pattern.encode())
        subclass_name = f"SymbolName_{xxh64.hexdigest()[-8:]}"
        namespace = dict(regex=regex)

        return type(subclass_name, (SymbolName,), namespace)

    def __repr__(self) -> str:
        return (
            f"SymbolName('{super().__repr__()}')"
            if type(self).__name__ == "SymbolName"
            else f"SymbolName.constrained('{self.regex.pattern}')('{super().__repr__()}')"
        )


class SymbolRef(ConstrainedStr):
    """Reference to a symbol name.

    Instance validation only happens automatically within a Pydantic
    model validation context.

    """

    @classmethod
    def from_string(cls, name: str) -> SymbolRef:
        name = cls.validate(name)
        return cls(name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"


class SourceLocation(pydantic.BaseModel):
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str
    end_line: Optional[PositiveInt]
    end_column: Optional[PositiveInt]

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
        super().__init__(
            line=line, column=column, source=source, end_line=end_line, end_column=end_column
        )

    def __str__(self) -> str:
        src = self.source or ""

        end_part = ""
        if self.end_line is not None:
            end_part += f" to Line {self.end_line}"
        if self.end_column is not None:
            end_part += f", Col {self.end_column}"

        return f"<'{src}': Line {self.line}, Col {self.column}{end_part}>"

    class Config:
        extra = "forbid"
        allow_mutation = False


class SourceLocationGroup(pydantic.BaseModel):
    """A group of merged source code locations (with optional info)."""

    locations: Tuple[SourceLocation, ...]
    context: Optional[Union[str, Tuple[str, ...]]]

    def __init__(
        self, *locations: SourceLocation, context: Optional[Union[str, Tuple[str, ...]]] = None
    ) -> None:
        super().__init__(locations=locations, context=context)

    def __str__(self) -> str:
        locs = ", ".join(str(loc) for loc in self.locations)
        context = f"#{self.context}#" if self.context else ""
        return f"<{context}[{locs}]>"

    @validator("locations")
    def non_empty_tuple(cls, v: Tuple[SourceLocation, ...]) -> Tuple[SourceLocation, ...]:
        if not v:
            raise ValueError("At least one location should be provided")
        return v
