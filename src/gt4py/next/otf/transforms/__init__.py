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

from gt4py.next.otf import stages, workflow

from .past_process_args import past_process_args
from .past_to_func import past_to_fun_def
from .past_to_itir import PastToItir, PastToItirFactory


__all__ = [
    "PastToItir",
    "PastToItirFactory",
    "past_to_fun_def",
    "past_process_args",
    "DEFAULT_TRANSFORMS",
]


DEFAULT_TRANSFORMS: workflow.Workflow[stages.PastClosure, stages.ProgramCall] = (
    past_process_args.chain(PastToItirFactory())
)
