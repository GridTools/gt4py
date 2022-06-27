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
from dataclasses import dataclass
from functools import update_wrapper
from typing import Protocol

from functional.iterator import ir as itir


class FencilFormatterFunction(Protocol):
    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        ...


class FencilExecutorFunction(Protocol):
    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        ...


@dataclass
class FencilFormatter:
    formatter_function: FencilFormatterFunction

    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        return self.formatter_function(fencil, *args, **kwargs)


@dataclass
class FencilExecutor:
    executor_function: FencilExecutorFunction

    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs) -> None:
        self.executor_function(fencil, *args, **kwargs)


def fencil_formatter(func: FencilFormatterFunction) -> FencilFormatter:
    wrapper = FencilFormatter(formatter_function=func)
    update_wrapper(wrapper, func)
    return wrapper


def fencil_executor(func: FencilExecutorFunction) -> FencilExecutor:
    wrapper = FencilExecutor(executor_function=func)
    update_wrapper(wrapper, func)
    return wrapper


def ensure_formatter(formatter: FencilFormatter) -> None:
    if not isinstance(formatter, FencilFormatter):
        raise RuntimeError(f"{formatter} is not a fencil formatter!")


def ensure_executor(executor: FencilExecutor) -> None:
    if not isinstance(executor, FencilExecutor):
        raise RuntimeError(f"{executor} is not a fencil executor!")
