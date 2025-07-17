# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.cartesian.gtc.cuir import cuir


class FuseKernels(eve.NodeTranslator):
    """Fuse consecutive kernels into single kernel for better launch overhead.

    Fuse kernels that are directly translated from OIR vertical loops into separate kernels if no synchronization is required.
    """

    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        def is_parallel(kernel: cuir.Kernel) -> bool:
            parallel = [
                loop.loop_order == cuir.LoopOrder.PARALLEL for loop in kernel.vertical_loops
            ]
            assert all(parallel) or not any(parallel), "Mixed k-parallelism in kernel"
            return any(parallel)

        kernels = [self.visit(node.kernels[0])]
        previous_parallel = is_parallel(kernels[-1])
        previous_writes = (
            kernels[-1]
            .walk_values()
            .if_isinstance(cuir.AssignStmt)
            .getattr("left")
            .if_isinstance(cuir.FieldAccess)
            .getattr("name")
            .to_set()
        )
        for kernel in node.kernels[1:]:
            parallel = is_parallel(kernel)
            reads_with_offsets = (
                kernel.walk_values()
                .if_isinstance(cuir.FieldAccess)
                .filter(
                    lambda x, parallel=parallel: any(
                        off != 0 for off in x.offset.to_dict().values()
                    )
                    or (x.offset.to_dict()["k"] != 0 and parallel)
                )
                .getattr("name")
                .to_set()
            )
            new_writes = (
                kernel.walk_values()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess)
                .getattr("name")
                .to_set()
            )

            if previous_parallel != parallel or reads_with_offsets & previous_writes:
                kernels.append(self.visit(kernel))
                previous_writes = new_writes
            else:
                kernels[-1].vertical_loops += self.visit(kernel.vertical_loops)
                previous_writes |= new_writes
            previous_parallel = parallel

        return cuir.Program(
            name=node.name,
            params=node.params,
            positionals=node.positionals,
            temporaries=node.temporaries,
            kernels=kernels,
        )
