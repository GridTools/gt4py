# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Set

from eve import NodeTranslator

from . import cuir


class FuseKernels(NodeTranslator):
    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        def is_parallel(kernel: cuir.Kernel) -> bool:
            parallel = [
                loop.loop_order == cuir.LoopOrder.PARALLEL for loop in kernel.vertical_loops
            ]
            assert all(parallel) or not any(parallel), "Mixed k-parallelism in kernel"
            return any(parallel)

        current_writes: Set[str] = set()
        kernels = [self.visit(node.kernels[0])]
        for kernel in node.kernels[1:]:
            reads_with_horizontal_offsets = (
                kernel.iter_tree()
                .if_isinstance(cuir.FieldAccess)
                .filter(lambda x: x.offset.i != 0 or x.offset.j != 0)
                .getattr("name")
                .to_set()
            )
            new_writes = (
                kernel.iter_tree()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess)
                .getattr("name")
                .to_set()
            )

            if (
                is_parallel(kernels[-1]) != is_parallel(kernel)
                or reads_with_horizontal_offsets & current_writes
            ):
                kernels.append(self.visit(kernel))
                current_writes = new_writes
            else:
                kernels[-1].vertical_loops += self.visit(kernel.vertical_loops)

        return cuir.Program(
            name=node.name, params=node.params, temporaries=node.temporaries, kernels=kernels
        )
