# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from __future__ import annotations

import functools
import inspect
import pathlib
import symtable
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast


MISSING_FILENAME = "<string>"


def get_closure_vars_from_function(function: Callable) -> dict[str, Any]:
    (nonlocals, globals, builtins, unbound) = inspect.getclosurevars(function)  # noqa: A001 [builtin-variable-shadowing]
    return {**builtins, **globals, **nonlocals}  # nonlocals override globals


def make_source_definition_from_function(func: Callable) -> SourceDefinition:
    try:
        filename = str(pathlib.Path(inspect.getabsfile(func)).resolve())
        if not filename:
            raise ValueError(
                "Can not create field operator from a function that is not in a source file."
            )
        source_lines, line_offset = inspect.getsourcelines(func)
        source_code = textwrap.dedent(inspect.getsource(func))
        column_offset = min(
            [len(line) - len(line.lstrip()) for line in source_lines if line.lstrip()], default=0
        )
        return SourceDefinition(source_code, filename, line_offset - 1, column_offset)

    except OSError as err:
        raise ValueError(f"Can not get source code of passed function '{func}'.") from err


def make_symbol_names_from_source(source: str, filename: str = MISSING_FILENAME) -> SymbolNames:
    try:
        mod_st = symtable.symtable(source, filename, "exec")
    except SyntaxError as err:
        raise ValueError(
            f"Unexpected error when parsing provided source code: \n{source}\n"
        ) from err

    assert mod_st.get_type() == "module"
    if len(children := mod_st.get_children()) != 1:
        raise ValueError(
            f"Sources with multiple function definitions are not yet supported: \n{source}\n"
        )

    assert children[0].get_type() == "function"
    func_st: symtable.Function = cast(symtable.Function, children[0])

    param_names: set[str] = set()
    imported_names: set[str] = set()
    local_names: set[str] = set()
    for name in func_st.get_locals():
        if (s := func_st.lookup(name)).is_imported():
            imported_names.add(name)
        elif s.is_parameter():
            param_names.add(name)
        else:
            local_names.add(name)

    # symtable returns regular free (or non-local) variables in 'get_frees()' and
    # the free variables introduced with the 'nonlocal' statement in 'get_nonlocals()'
    nonlocal_names = set(func_st.get_frees()) | set(func_st.get_nonlocals())
    global_names = set(func_st.get_globals())

    return SymbolNames(
        params=param_names,
        locals=local_names,
        imported=imported_names,
        nonlocals=nonlocal_names,
        globals=global_names,
    )


@dataclass(frozen=True)
class SourceDefinition:
    """
    A GT4Py source code definition encoded as a string.

    It can be created from an actual function object using :meth:`from_function()`.
    It also supports unpacking.


    Examples
    --------
    >>> def foo(a):
    ...     return a
    >>> src_def = SourceDefinition.from_function(foo)
    >>> print(src_def)  # doctest:+ELLIPSIS
    SourceDefinition(source='def foo(a):...', filename='...', line_offset=0, column_offset=0)

    >>> source, filename, starting_line = src_def
    >>> print(source)  # doctest:+ELLIPSIS
    def foo(a):
        return a
    ...
    """

    source: str
    filename: str = MISSING_FILENAME
    line_offset: int = 0
    column_offset: int = 0

    def __iter__(self) -> Iterator:
        yield self.source
        yield self.filename
        yield self.line_offset

    from_function = staticmethod(make_source_definition_from_function)


@dataclass(frozen=True)
class SymbolNames:
    """
    Collection of symbol names used in a function classified by kind.

    It can be created directly from source code using :meth:`from_source()`.
    It also supports unpacking.
    """

    params: set[str]
    locals: set[str]  # shadowing a python builtin
    imported: set[str]
    nonlocals: set[str]
    globals: set[str]  # shadowing a python builtin

    @functools.cached_property
    def all_locals(self) -> set[str]:
        return self.params | self.locals | self.imported

    def __iter__(self) -> Iterator[set[str]]:
        yield self.params
        yield self.locals
        yield self.imported
        yield self.nonlocals
        yield self.globals

    from_source = staticmethod(make_symbol_names_from_source)
