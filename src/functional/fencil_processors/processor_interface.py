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


class ProcessorType(enum.Enum):
    EXECUTOR = 0
    FORMATTER = 1


class Processor(typing.Protocol):
    processor_type: ProcessorType

    def __call__(fencil: itir.FencilDefinition, *args, **kwargs) -> typing.Any:
        ...


def fencil_formatter(func):
    func.processor_type = ProcessorType.FORMATTER
    return func


def fencil_executor(func):
    func.processor_type = ProcessorType.EXECUTOR
    return func


def ensure_formatter(formatter: Processor) -> None:
    if (
        not hasattr(formatter, "processor_type")
        or formatter.processor_type is not ProcessorType.FORMATTER
    ):
        raise RuntimeError(f"{formatter} is not marked as a fencil formatter!")


def ensure_executor(backend: Processor) -> None:
    if (
        not hasattr(backend, "processor_type")
        or backend.processor_type is not ProcessorType.EXECUTOR
    ):
        raise RuntimeError(f"{backend} is not marked as a fencil executor!")
