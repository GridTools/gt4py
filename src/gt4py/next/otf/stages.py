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

import dataclasses
from typing import Any, Generic, Iterable, Iterator, Optional, Protocol, TypeVar

import numpy as np

from gt4py.eve.extended_typing import Self
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_specifications as ts, type_translation


SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
SettingT = TypeVar("SettingT", bound=languages.LanguageSettings)
SrcL_co = TypeVar("SrcL_co", bound=languages.LanguageTag, covariant=True)
TgtL_co = TypeVar("TgtL_co", bound=languages.LanguageTag, covariant=True)
SettingT_co = TypeVar("SettingT_co", bound=languages.LanguageSettings, covariant=True)


@dataclasses.dataclass(frozen=True)
class CompileArg:
    gt_type: ts.TypeSpec

    def __gt_type__(self) -> ts.TypeSpec:
        return self.gt_type


@dataclasses.dataclass(frozen=True)
class CompileConnectivity(common.Connectivity):
    max_neighbors: int
    has_skip_values: bool
    origin_axis: common.Dimension
    neighbor_axis: common.Dimension
    index_type: type[int] | type[np.int32] | type[np.int64]

    @classmethod
    def from_connectivity(cls, connectivity: common.Connectivity) -> Self:
        return cls(
            max_neighbors=connectivity.max_neighbors,
            has_skip_values=connectivity.has_skip_values,
            origin_axis=connectivity.origin_axis,
            neighbor_axis=connectivity.neighbor_axis,
            index_type=connectivity.index_type,
        )


def connectivity_or_dimension(
    some_offset_provider: common.Connectivity | common.Dimension,
) -> CompileConnectivity | common.Dimension:
    match some_offset_provider:
        case common.Dimension():
            return some_offset_provider
        case common.Connectivity():
            return CompileConnectivity.from_connectivity(some_offset_provider)
        case _:
            raise ValueError


def iter_size_compile_args(args: Iterable[CompileArg]) -> Iterator[CompileArg]:
    for arg in args:
        match argt := arg.__gt_type__():
            case ts.TupleType():
                yield from iter_size_compile_args((CompileArg(t) for t in argt))
            case ts.FieldType():
                yield from [
                    CompileArg(ts.ScalarType(kind=ts.ScalarKind.INT32)) for dim in argt.dims
                ]
            case _:
                pass


@dataclasses.dataclass(frozen=True)
class CompileArgSpec:
    args: tuple[CompileArg, ...]
    kwargs: dict[str, CompileArg]
    offset_provider: dict[str, common.Connectivity | common.Dimension]
    column_axis: Optional[common.Dimension]

    @classmethod
    def from_concrete_no_size(cls, *args: Any, **kwargs: Any) -> Self:
        compile_args = tuple(CompileArg(type_translation.from_value(arg)) for arg in args)
        kwargs_copy = kwargs.copy()
        return cls(
            args=compile_args,
            offset_provider=kwargs_copy.pop("offset_provider", {}),
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: CompileArg(type_translation.from_value(v))
                for k, v in kwargs_copy.items()
                if v is not None
            },
        )

    @classmethod
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
        compile_args = tuple(CompileArg(type_translation.from_value(arg)) for arg in args)
        size_args = tuple(iter_size_compile_args(compile_args))
        kwargs_copy = kwargs.copy()
        return cls(
            args=(*compile_args, *size_args),
            offset_provider=kwargs_copy.pop("offset_provider", {}),
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: CompileArg(type_translation.from_value(v))
                for k, v in kwargs_copy.items()
                if v is not None
            },
        )


@dataclasses.dataclass(frozen=True)
class JITArgs:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class AOTProgram:
    program: itir.FencilDefinition
    argspec: CompileArgSpec


@dataclasses.dataclass(frozen=True)
class ProgramCall:
    """Iterator IR representaion of a program together with arguments to be passed to it."""

    program: itir.FencilDefinition
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def program_call_to_aot_program(program_call: ProgramCall) -> AOTProgram:
    return AOTProgram(
        program=program_call.program,
        argspec=CompileArgSpec.from_concrete(*program_call.args, **program_call.kwargs),
    )


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
