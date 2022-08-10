# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


from eve.concepts import Node
from functional.fencil_processors.processor_interface import fencil_executor
from functional.fencil_processors.runners import roundtrip


@fencil_executor
def executor(ir: Node, *args, **kwargs):
    roundtrip.executor(ir, *args, dispatch_backend=roundtrip.executor, **kwargs)
