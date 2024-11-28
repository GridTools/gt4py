# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc.common import LoopOrder
from gt4py.cartesian.gtc.cuir import kernel_fusion

from .cuir_utils import (
    AssignStmtFactory,
    KernelFactory,
    ProgramFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_forward_backward_fusion():
    testee = ProgramFactory(
        kernels=[
            KernelFactory(
                vertical_loops__0=VerticalLoopFactory(
                    loop_order=LoopOrder.FORWARD,
                    sections=[
                        VerticalLoopSectionFactory(
                            end__level=0,
                            end__offset=1,
                            horizontal_executions__0__body__0=AssignStmtFactory(
                                left__name="tmp", right__name="inp"
                            ),
                        ),
                        VerticalLoopSectionFactory(
                            start__offset=1,
                            horizontal_executions__0__body__0=AssignStmtFactory(
                                left__name="tmp", right__name="tmp", right__offset__k=-1
                            ),
                        ),
                    ],
                )
            ),
            KernelFactory(
                vertical_loops__0=VerticalLoopFactory(
                    loop_order=LoopOrder.BACKWARD,
                    sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="out", right__name="tmp"
                    ),
                )
            ),
        ]
    )
    transformed = kernel_fusion.FuseKernels().visit(testee)
    assert len(transformed.kernels) == 1
    assert transformed.kernels[0].vertical_loops == (
        testee.kernels[0].vertical_loops + testee.kernels[1].vertical_loops
    )


def test_no_fusion_with_parallel_offsets():
    testee = ProgramFactory(
        kernels=[
            KernelFactory(
                vertical_loops__0=VerticalLoopFactory(
                    loop_order=LoopOrder.FORWARD,
                    sections=[
                        VerticalLoopSectionFactory(
                            end__level=0,
                            end__offset=1,
                            horizontal_executions__0__body__0=AssignStmtFactory(
                                left__name="tmp", right__name="inp"
                            ),
                        ),
                        VerticalLoopSectionFactory(
                            start__offset=1,
                            horizontal_executions__0__body__0=AssignStmtFactory(
                                left__name="tmp", right__name="tmp", right__offset__k=-1
                            ),
                        ),
                    ],
                )
            ),
            KernelFactory(
                vertical_loops__0=VerticalLoopFactory(
                    loop_order=LoopOrder.BACKWARD,
                    sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                        left__name="out", right__name="tmp", right__offset__i=1
                    ),
                )
            ),
        ]
    )
    transformed = kernel_fusion.FuseKernels().visit(testee)
    assert len(transformed.kernels) == 2

    testee = ProgramFactory(
        kernels=[
            KernelFactory(
                vertical_loops__0__sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                    left__name="tmp", right__name="inp"
                )
            ),
            KernelFactory(
                vertical_loops__0__sections__0__horizontal_executions__0__body__0=AssignStmtFactory(
                    left__name="out", right__name="tmp", right__offset__k=1
                )
            ),
        ]
    )
    transformed = kernel_fusion.FuseKernels().visit(testee)
    assert len(transformed.kernels) == 2
