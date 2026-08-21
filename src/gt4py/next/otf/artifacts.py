# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data models of the artifacts flowing through a build pipeline.

This module hosts the DSL-agnostic vocabulary of on-the-fly compilation:
source code specifications (`SourceCodeSpec` and friends, ex `code_specs`
module), the source containers produced and consumed by the pipeline steps
(`ProgramSource`, `BindingSource`, `ExtensionSource`) and the contracts of
its end products (`CompilationArtifact`, `ExecutableProgram`,
`BuildSystemProject`). Nothing in here knows about GT4Py IRs or the type
system; what a program *is* stays in `gt4py.next.otf.stages`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any, Generic, Optional, Protocol, TypeAlias, TypeVar, runtime_checkable

from gt4py.eve import codegen
from gt4py.next.otf.binding import interface


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceCodeSpec:
    """
    Basic settings for any source programming language.

    Formatting will happen through ``eve.codegen.format_source``.
    For available formatting options, check the options of the
    specific formatter used depending on ``.formatter_key``.
    """

    source_language: str
    file_extension: str
    formatter_key: str | None = None
    formatter_options: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class HeaderAndSourceCodeSpec(SourceCodeSpec):
    """Add a header file extension setting on top of the basic set."""

    header_extension: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class PythonCodeSpec(SourceCodeSpec):
    """Settings for Python language."""

    source_language: str = "python"
    file_extension: str = "py"
    formatter_key: str = "python"


@dataclasses.dataclass(frozen=True, kw_only=True)
class SDFGCodeSpec(SourceCodeSpec):
    """Settings for SDFGs."""

    source_language: str = "SDFG"
    file_extension: str = "sdfg"


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPPLikeCodeSpec(HeaderAndSourceCodeSpec):
    """Settings for C++-like language."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPPCodeSpec(CPPLikeCodeSpec):
    """Settings for C++ language."""

    source_language: str = "CXX"
    file_extension: str = "cpp"
    header_extension: str = "hpp"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CUDACodeSpec(CPPLikeCodeSpec):
    """Settings for CUDA language."""

    source_language: str = "CUDA"
    file_extension: str = "cu"
    header_extension: str = "cuh"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HIPCodeSpec(CPPLikeCodeSpec):
    """Settings for HIP language."""

    source_language: str = "HIP"
    file_extension: str = "hip"
    header_extension: str = "h"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, style="LLVM")
    )


def format_source(source_code_spec: SourceCodeSpec, source: str) -> str:
    assert source_code_spec.formatter_key is not None, (
        "No formatter key specified in source code specification."
    )
    return codegen.format_source(
        source_code_spec.formatter_key, source, **(source_code_spec.formatter_options or {})
    )


CodeSpecT = TypeVar("CodeSpecT", bound=SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=SourceCodeSpec)


@dataclasses.dataclass(frozen=True)
class ProgramSource(Generic[CodeSpecT]):
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
    code_spec: CodeSpecT


@dataclasses.dataclass(frozen=True)
class BindingSource(Generic[CodeSpecT, TargetCodeSpecT]):
    """
    Companion source code for translated program source code.

    This is only needed for OTF compilation if the translated program source code is
    not directly callable from python and therefore requires bindings.
    This can also optionally be added to compile bindings for other languages than python
    when using GT4Py as part of the build for a non-python driver project.
    """

    source_code: str
    library_deps: tuple[interface.LibraryDependency, ...]


@dataclasses.dataclass(frozen=True)
class ExtensionSource(Generic[CodeSpecT, TargetCodeSpecT]):
    """
    Encapsulate all the source code required for OTF compilation.

    The bindings module is optional if and only if the program_source is directly callable.
    This should only be the case if the source language / framework supports this out of the box.
    If bindings are required, it is recommended to create them in a separate step to ensure reusability.
    """

    program_source: ProgramSource[CodeSpecT]
    binding_source: Optional[BindingSource[CodeSpecT, TargetCodeSpecT]]

    @property
    def library_deps(self) -> tuple[interface.LibraryDependency, ...]:
        if not self.binding_source:
            return self.program_source.library_deps
        return _unique_libs(*self.program_source.library_deps, *self.binding_source.library_deps)


CodeSpecT_co = TypeVar("CodeSpecT_co", bound=SourceCodeSpec, covariant=True)
TargetCodeSpecT_co = TypeVar("TargetCodeSpecT_co", bound=SourceCodeSpec, covariant=True)


class BuildSystemProject(Protocol[CodeSpecT_co, TargetCodeSpecT_co]):
    """
    Use source code extracted from an ``ExtensionSource`` to configure and build a GT4Py program.

    Should only be considered an OTF stage if used as an endpoint, as this only runs commands on source files
    and is not responsible for importing the results into Python.
    """

    def build(self) -> None: ...


ExecutableProgram: TypeAlias = Callable


@runtime_checkable
class CompilationArtifact(Protocol):
    """The output of an ``OTFCompileWorkflow``.

    Each backend defines its own concrete artifact dataclass; all share this
    Protocol. Implementations are frozen dataclasses, picklable, and carry no
    live process-bound state — that is reconstructed by ``load``, which
    returns a directly-callable ``ExecutableProgram`` taking gt4py-shaped
    arguments.

    The one current exception is ``RoundtripArtifact`` when it is configured
    with a ``dispatch_backend``: that field holds a ``Backend`` reference
    whose role belongs at the runner / load-time seam, not in the artifact
    itself.
    """

    def load(self) -> ExecutableProgram: ...


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
