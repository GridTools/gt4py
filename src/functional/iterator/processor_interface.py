# GT4Py Project - GridTools Framework
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
import enum
import typing

from functional.iterator import ir as itir


class ProcessorKind(enum.Enum):
    EXECUTOR = 0
    FORMATTER = 1


class Executor(typing.Protocol):
    processor_kind: ProcessorKind

    def __call__(fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        ...


class Formatter(typing.Protocol):
    processor_kind: ProcessorKind

    def __call__(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        ...


def fencil_formatter(func: typing.Callable) -> Formatter:
    func.processor_kind = ProcessorKind.FORMATTER  # type: ignore[attr-defined]  # we want to add the attr if necessary
    return func  # type: ignore[return-value]  # mypy does not recognize compatibility of this protocol


def fencil_executor(func: typing.Callable) -> Executor:
    func.processor_kind = ProcessorKind.EXECUTOR  # type: ignore[attr-defined]  # we want to add the attr if necessary
    return func  # type: ignore[return-value]  # mypy does not recognize compatibility of this protocol


def ensure_formatter(formatter: Executor) -> None:
    if (
        not hasattr(formatter, "processor_kind")
        or formatter.processor_kind is not ProcessorKind.FORMATTER
    ):
        raise RuntimeError(f"{formatter} is not marked as a fencil formatter!")


def ensure_executor(backend: Formatter) -> None:
    if (
        not hasattr(backend, "processor_kind")
        or backend.processor_kind is not ProcessorKind.EXECUTOR
    ):
        raise RuntimeError(f"{backend} is not marked as a fencil executor!")
