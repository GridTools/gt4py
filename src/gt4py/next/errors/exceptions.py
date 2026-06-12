# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The list of exception classes used in the library.

Exception classes that represent errors within an IR go here as a subclass of
:class:`DSLError`. Exception classes that represent other errors, like
the builtin ValueError, go here as well, although you should use Python's
builtin error classes if you can. Exception classes that are specific to a
certain submodule and have no use for the entire application may be better off
in that submodule as opposed to being in this file.
"""

from __future__ import annotations

import difflib
from typing import Any, ClassVar, Iterable, Optional, Sequence

from gt4py.eve import SourceLocation

# TODO(havogt): import 'Self' from 'typing' directly once the Python floor is >=3.12.
from gt4py.eve.extended_typing import Self
from gt4py.next.errors import formatting


class GT4PyError(Exception):
    @property
    def message(self) -> str:
        return self.args[0]


def _did_you_mean(name: str, candidates: Iterable[str]) -> list[str]:
    """Produce a 'Did you mean ...?' hint if `name` closely matches any candidate."""
    # Never suggest the name the user already wrote (it can appear among the
    # candidates as a same-named symbol from another SSA generation).
    matches = difflib.get_close_matches(name, [c for c in candidates if c != name], n=3, cutoff=0.6)
    if not matches:
        return []
    return [f"Did you mean {' or '.join(f'{m!r}' for m in matches)}?"]


class DSLError(GT4PyError):
    """
    Error in user code of one of the GT4Py-embedded DSLs.

    Besides the message and the primary source location, a diagnostic can carry
    optional structured payload that the formatter renders for the user:

    - ``label``: short text printed right after the carets, qualifying the
      marked code (e.g. "this has type 'bool'").
    - ``related``: further (location, message) pairs that contribute to the
      error (e.g. the other operand of a type mismatch).
    - ``notes``: factual background ("Note: ..."), explaining *why* this is an
      error.
    - ``hints``: actionable advice ("Hint: ..."), telling the user what to do
      instead.
    - ``code``: a stable, machine-readable identifier of the error category,
      set per subclass; intended for searching documentation and for tooling.
    """

    code: ClassVar[Optional[str]] = None

    location: Optional[SourceLocation]
    label: Optional[str]
    related: list[tuple[SourceLocation, str]]
    notes: list[str]
    hints: list[str]

    def __init__(
        self,
        location: Optional[SourceLocation],
        message: str,
        *,
        label: Optional[str] = None,
        related: Sequence[tuple[SourceLocation, str]] = (),
        notes: Sequence[str] = (),
        hints: Sequence[str] = (),
    ) -> None:
        self.location = location
        self.label = label
        self.related = list(related)
        self.notes = list(notes)
        self.hints = list(hints)
        super().__init__(message)

    def with_location(self, location: Optional[SourceLocation]) -> Self:
        self.location = location
        return self

    # TODO(havogt): on Python >=3.11 this shadows 'BaseException.add_note' (PEP 678);
    #  on 3.10 it is a plain new method, so 'add_note' on GT4Py exceptions that are
    #  not 'DSLError's raises 'AttributeError' on 3.10 but silently goes to
    #  '__notes__' on 3.11+. Remove this caveat once the Python floor is >=3.12.
    def add_note(self, note: str) -> None:
        """
        Add a note to the diagnostic, using the standard 'BaseException.add_note' API.

        Toolchain stages use this to attach context to an in-flight error
        ("While processing ...") without touching the message.

        The note is routed into the structured 'notes' field instead of
        '__notes__': the traceback machinery (and IPython/pytest) renders this
        exception through 'str()', which already includes the structured notes,
        so storing them in '__notes__' as well would print them twice.
        """
        self.notes.append(note)

    def __str__(self) -> str:
        return formatting.format_diagnostic_parts(
            self.message,
            self.location,
            label=self.label,
            related=self.related,
            notes=self.notes,
            hints=self.hints,
        )


class UnsupportedPythonFeatureError(DSLError):
    code = "unsupported-syntax"

    feature: str

    def __init__(
        self,
        location: Optional[SourceLocation],
        feature: str,
        *,
        notes: Sequence[str] = (),
        hints: Sequence[str] = (),
    ) -> None:
        super().__init__(
            location, f"Unsupported Python syntax: {feature}.", notes=notes, hints=hints
        )
        self.feature = feature


class UndefinedSymbolError(DSLError):
    code = "undefined-symbol"

    sym_name: str

    def __init__(
        self,
        location: Optional[SourceLocation],
        name: str,
        *,
        candidates: Iterable[str] = (),
    ) -> None:
        super().__init__(
            location,
            f"Undeclared symbol '{name}'.",
            label="not defined at this point",
            hints=_did_you_mean(name, candidates),
        )
        self.sym_name = name


class MissingAttributeError(DSLError):
    attr_name: str

    def __init__(self, location: Optional[SourceLocation], attr_name: str) -> None:
        super().__init__(location, f"Object does not have attribute '{attr_name}'.")
        self.attr_name = attr_name


class MissingArgumentError(DSLError):
    arg_name: str
    is_kwarg: bool

    def __init__(self, location: Optional[SourceLocation], arg_name: str, is_kwarg: bool) -> None:
        super().__init__(
            location, f"Expected {'keyword-' if is_kwarg else ''}argument '{arg_name}'."
        )
        self.attr_name = arg_name
        self.is_kwarg = is_kwarg


class DSLTypeError(DSLError):
    def __init__(self, location: Optional[SourceLocation], message: str) -> None:
        super().__init__(location, message)


class MissingParameterAnnotationError(DSLTypeError):
    param_name: str

    def __init__(self, location: Optional[SourceLocation], param_name: str) -> None:
        super().__init__(location, f"Parameter '{param_name}' is missing type annotations.")
        self.param_name = param_name


class InvalidParameterAnnotationError(DSLTypeError):
    param_name: str
    annotated_type: Any

    def __init__(self, location: Optional[SourceLocation], param_name: str, type_: Any) -> None:
        super().__init__(
            location, f"Parameter '{param_name}' has invalid type annotation '{type_}'."
        )
        self.param_name = param_name
        self.annotated_type = type_


class CompilationError(GT4PyError):
    def __init__(self, compilation_error: str) -> None:
        super().__init__(f"See attached compilation log.\n{compilation_error}")


class EmbeddedExecutionError(GT4PyError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
