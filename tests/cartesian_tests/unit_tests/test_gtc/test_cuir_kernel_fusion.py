# GT4Py - GridTools Framework
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
        ],
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
        ],
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
        ],
    )
    transformed = kernel_fusion.FuseKernels().visit(testee)
    assert len(transformed.kernels) == 2
