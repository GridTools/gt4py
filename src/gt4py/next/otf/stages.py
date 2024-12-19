# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Generic, Optional, Protocol, TypeAlias, TypeVar

from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, languages, toolchain
from gt4py.next.otf.binding import interface


PrgT = TypeVar("PrgT")
ArgT = TypeVar("ArgT")
SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
SettingT = TypeVar("SettingT", bound=languages.LanguageSettings)
SrcL_co = TypeVar("SrcL_co", bound=languages.LanguageTag, covariant=True)
TgtL_co = TypeVar("TgtL_co", bound=languages.LanguageTag, covariant=True)
SettingT_co = TypeVar("SettingT_co", bound=languages.LanguageSettings, covariant=True)


CompilableProgram: TypeAlias = toolchain.CompilableProgram[itir.Program, arguments.CompileTimeArgs]


@dataclasses.dataclass(frozen=True)
class ProgramSource(Generic[SrcL, SettingT]):
    """
    Standalone source code translated from an IR along with information relevant for OTF compilation.

    Contains additional information required for further OTF steps, such as
    - implementation language and language conventions
    - dependencies on implementation language libraries
    - how to call the program
    """

    entry_point: interface.Function
    source_code: str
    library_deps: tuple[interface.LibraryDependency, ...]
    language: type[SrcL]
    language_settings: SettingT
    implicit_domain: bool

    def __post_init__(self) -> None:
        if not isinstance(self.language_settings, self.language.settings_class):
            raise TypeError(
                f"Wrong language settings type for '{self.language}', must be subclass of '{self.language.settings_class}'."
            )


@dataclasses.dataclass(frozen=True)
class BindingSource(Generic[SrcL, TgtL]):
    """
    Companion source code for translated program source code.

    This is only needed for OTF compilation if the translated program source code is
    not directly callable from python and therefore requires bindings.
    This can also optionally be added to compile bindings for other languages than python
    when using GT4Py as part of the build for a non-python driver project.
    """

    source_code: str
    library_deps: tuple[interface.LibraryDependency, ...]


# TODO(ricoh): reconsider name in view of future backends producing standalone compilable ProgramSource code
@dataclasses.dataclass(frozen=True)
class CompilableSource(Generic[SrcL, SettingT, TgtL]):
    """
    Encapsulate all the source code required for OTF compilation.

    The bindings module is optional if and only if the program_source is directly callable.
    This should only be the case if the source language / framework supports this out of the box.
    If bindings are required, it is recommended to create them in a separate step to ensure reusability.
    """

    program_source: ProgramSource[SrcL, SettingT]
    binding_source: Optional[BindingSource[SrcL, TgtL]]

    @property
    def library_deps(self) -> tuple[interface.LibraryDependency, ...]:
        if not self.binding_source:
            return self.program_source.library_deps
        return _unique_libs(*self.program_source.library_deps, *self.binding_source.library_deps)


class BuildSystemProject(Protocol[SrcL_co, SettingT_co, TgtL_co]):
    """
    Use source code extracted from a ``CompilableSource`` to configure and build a GT4Py program.

    Should only be considered an OTF stage if used as an endpoint, as this only runs commands on source files
    and is not responsible for importing the results into Python.
    """

    def build(self) -> None: ...


class CompiledProgram(Protocol):
    """Executable python representation of a program."""

    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class ExtendedCompiledProgram(CompiledProgram):
    """Executable python representation of a program with extra info."""

    implicit_domain: bool


def _unique_libs(*args: interface.LibraryDependency) -> tuple[interface.LibraryDependency, ...]:
    """
    Filter out multiple occurrences of the same ``interface.LibraryDependency``.

    Examples:
    ---------
    >>> libs_a = (
    ...     interface.LibraryDependency("foo", "1.2.3"),
    ...     interface.LibraryDependency("common", "1.0.0"),
    ... )
    >>> libs_b = (
    ...     interface.LibraryDependency("common", "1.0.0"),
    ...     interface.LibraryDependency("bar", "1.2.3"),
    ... )
    >>> _unique_libs(*libs_a, *libs_b)
    (LibraryDependency(name='foo', version='1.2.3'), LibraryDependency(name='common', version='1.0.0'), LibraryDependency(name='bar', version='1.2.3'))
    """
    unique: list[interface.LibraryDependency] = []
    for lib in args:
        if lib not in unique:
            unique.append(lib)
    return tuple(unique)
