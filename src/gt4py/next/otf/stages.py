# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Generic, Optional, Protocol, TypeAlias, TypeVar

from gt4py.eve import utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import code_specs, definitions
from gt4py.next.otf.binding import interface


def compilation_hash(program_def: definitions.CompilableProgramDef) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = program_def.args.offset_provider
    return hash(
        (
            program_def.data,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            utils.content_hash(tuple(arg for arg in program_def.args.args)),
            common.hash_offset_provider_items_by_id(offset_provider) if offset_provider else None,
            program_def.args.column_axis,
        )
    )


def fingerprint_compilable_program(program_def: definitions.CompilableProgramDef) -> str:
    """
    Generates a unique hash string for a stencil source program representing
    the program, sorted offset_provider, and column_axis.
    """
    program: itir.Program = program_def.data
    offset_provider: common.OffsetProvider = program_def.args.offset_provider
    column_axis: Optional[common.Dimension] = program_def.args.column_axis

    program_hash = utils.content_hash(
        (
            program.fingerprint(),
            sorted(offset_provider.items(), key=lambda el: el[0]),
            column_axis,
        )
    )

    return program_hash


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
ToCodeSpecT = TypeVar("ToCodeSpecT", bound=code_specs.SourceCodeSpec)


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
class BindingSource(Generic[CodeSpecT, ToCodeSpecT]):
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
class CompilableProject(Generic[CodeSpecT, ToCodeSpecT]):
    """
    Encapsulate all the source code required for OTF compilation.

    The bindings module is optional if and only if the program_source is directly callable.
    This should only be the case if the source language / framework supports this out of the box.
    If bindings are required, it is recommended to create them in a separate step to ensure reusability.
    """

    program_source: ProgramSource[CodeSpecT]
    binding_source: Optional[BindingSource[CodeSpecT, ToCodeSpecT]]

    @property
    def library_deps(self) -> tuple[interface.LibraryDependency, ...]:
        if not self.binding_source:
            return self.program_source.library_deps
        return _unique_libs(*self.program_source.library_deps, *self.binding_source.library_deps)


CodeSpecT_co = TypeVar("CodeSpecT_co", bound=code_specs.SourceCodeSpec, covariant=True)
ToCodeSpecT_co = TypeVar("ToCodeSpecT_co", bound=code_specs.SourceCodeSpec, covariant=True)


class BuildSystemProject(Protocol[CodeSpecT_co, ToCodeSpecT_co]):
    """
    Use source code extracted from a ``CompilableSource`` to configure and build a GT4Py program.

    Should only be considered an OTF stage if used as an endpoint, as this only runs commands on source files
    and is not responsible for importing the results into Python.
    """

    def build(self) -> None: ...


ExecutableProgram: TypeAlias = Callable


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
