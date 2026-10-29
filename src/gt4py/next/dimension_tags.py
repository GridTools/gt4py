# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Dimension tags: their spelling in generated code and their resolution back to classes.

A dimension's tag is its qualified Python name, `module:qualname` (see ADR 0029). This module
holds the two translations of a tag that live outside the dimension model itself:

- `codegen_name` / `from_codegen_name`: the injective mapping between a tag (or an offset
  tag) and a valid identifier in generated code, shared by all backends.
- `resolve`: the dimension class a tag names, by importing it, for code running embedded in
  the user's process; `bind_dimension_classes` pins the answer to a program's own classes.

The public entry point `resolve` is re-exported as `gtx.resolve`.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import re
from collections.abc import Iterator, Mapping
from typing import Any, Final, Optional, cast

from gt4py.next import common
from gt4py.next.common import Dimension, DimensionRef, Tag


_CODEGEN_UNESCAPE: Final = {"u": "_", "d": ".", "c": ":", "l": "[", "r": "]"}


def codegen_name(tag: Tag) -> str:
    """
    Mangle a dimension or offset tag into a valid generated identifier.

    A tag contains dots, a colon and, for a parametrized dimension such as `Staggered[pkg:K]`,
    brackets, all illegal in a C++ identifier, a DaCe symbol and `eve`'s `SymbolName`
    (`^[a-zA-Z_]\\w*$`). The underscore is the only available separator, so it is escaped
    too: every `_` in the result starts a two-character escape, which keeps the mapping
    injective (unlike, e.g., doubling underscores and mapping dots to single ones, where
    `'..'` and `'_'` would collide).

    Args:
        tag: A dimension or offset tag, i.e. a qualified Python name.

    Returns:
        A valid identifier, unique for each distinct `tag`.

    Examples:
        >>> codegen_name("mod:V2E.Local")
        'mod_cV2E_dLocal'
        >>> codegen_name("a_b.c")
        'a_ub_dc'
        >>> from_codegen_name(codegen_name("my__mod:X"))
        'my__mod:X'
    """
    # `_` first, so the underscores introduced by the other escapes are not re-escaped
    return (
        tag.replace("_", "_u")
        .replace(".", "_d")
        .replace(":", "_c")
        .replace("[", "_l")
        .replace("]", "_r")
    )


def from_codegen_name(name: str) -> Tag:
    """
    Recover a tag from the identifier `codegen_name` produced for it.

    Args:
        name: An identifier produced by `codegen_name`.

    Returns:
        The original tag.

    Examples:
        >>> from_codegen_name("mod_cV2E_dLocal")
        'mod:V2E.Local'
    """
    return re.sub(r"_([udclr])", lambda m: _CODEGEN_UNESCAPE[m.group(1)], name)


#: Set while `bind_dimension_classes` is active: tag -> the class `resolve` must answer with.
_DIMENSION_CLASS_BINDINGS: contextvars.ContextVar[Optional[dict[Tag, Dimension]]] = (
    contextvars.ContextVar("_DIMENSION_CLASS_BINDINGS", default=None)
)


@contextlib.contextmanager
def bind_dimension_classes(classes: Mapping[DimensionRef, Dimension]) -> Iterator[None]:
    """
    Make `resolve` answer with these classes (e.g. a program's own declarations) for their tags.

    A program loaded after its dimensions were redefined (a re-run notebook cell) must run on
    the classes it was declared with, not on whatever its tags import now. Other tags still
    resolve by import.
    """
    token = _DIMENSION_CLASS_BINDINGS.set({ref.tag: cls for ref, cls in classes.items()})
    try:
        yield
    finally:
        _DIMENSION_CLASS_BINDINGS.reset(token)


def resolve(dim: Tag | DimensionRef) -> Dimension:
    """
    Return the dimension class a tag or ref names, by importing it.

    Not memoized: the answer is the module's *current* attribute, so a dimension redefined in
    a re-run notebook cell resolves to the new class (`importlib` already caches the module).

    Only for code running in the user's process that needs the class back: iterator-embedded
    execution and the roundtrip backend. The compile path works on `DimensionRef` values.

    A tag is `module:qualname` (`pkgutil.resolve_name` syntax), so this is an import followed by
    an attribute walk -- the same way `pickle` references a class. Parametrized dimensions such
    as `Staggered[K]` have no importable qualname; their tag has the form
    `<owner tag>[<base tag>]` and is resolved by subscripting the owner, which goes through its
    intern table and so returns the identical class.

    Args:
        dim: A dimension tag, or a ref (whose kind and category must then match the class).

    Returns:
        The dimension class.

    Raises:
        ValueError: If the module is not importable, the attribute walk fails, or a given ref
            does not describe the class found (e.g. it was redefined with another kind).

    Examples:
        >>> resolve("gt4py.next.common:DimensionIndex") is common.DimensionIndex
        True
    """
    if isinstance(dim, DimensionRef):
        cls = resolve(dim.tag)
        if (found := common.dimension_ref(cls)) != dim:
            raise ValueError(
                f"'{dim.tag}' resolves to '{found!r}', which does not match '{dim!r}'."
            )
        return cls

    tag = dim
    if (bindings := _DIMENSION_CLASS_BINDINGS.get()) is not None and (
        bound := bindings.get(tag)
    ) is not None:
        return bound
    if (match := common._STAGGERED_TAG_RE.match(tag)) is not None:
        owner = resolve(match["owner"])
        if not isinstance(owner, common.StaggeredMeta):
            raise ValueError(
                f"Cannot resolve tag '{tag}': '{match['owner']}' is not a parametrized dimension."
            )
        return owner[resolve(match["base"])]  # type: ignore[index] # a StaggeredMeta, checked

    module_name, sep, qualname = tag.partition(":")
    if not sep:
        raise ValueError(f"'{tag}' is not a dimension tag ('module:qualname').")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as ex:
        raise ValueError(
            f"Cannot resolve dimension tag '{tag}': module '{module_name}' is not importable. A"
            " dimension referenced from the IR must be declared at module level in an importable"
            " module."
        ) from ex
    for attr in qualname.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as ex:
            raise ValueError(
                f"Cannot resolve dimension tag '{tag}': '{module_name}' has no attribute '{qualname}'."
            ) from ex
    if not isinstance(obj, common.DimensionMeta):
        raise ValueError(f"Tag '{tag}' resolves to '{obj}', which is not a dimension.")
    return cast(Dimension, obj)
