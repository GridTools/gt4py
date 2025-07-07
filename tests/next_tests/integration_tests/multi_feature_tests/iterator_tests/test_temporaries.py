# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.builtins import (
    as_fieldop,
    cartesian_domain,
    deref,
    float64,
    named_range,
    shift,
)
from gt4py.next.iterator.runtime import fendef, fundef, offset, set_at, temporary

from next_tests.unit_tests.conftest import program_processor_no_transforms, run_processor


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return (
        -4.0 * deref(inp)
        + deref(shift(i, 1)(inp))
        + deref(shift(i, -1)(inp))
        + deref(shift(j, 1)(inp))
        + deref(shift(j, -1)(inp))
    )


def lap_ref(inp):
    return -4.0 * inp[1:-1, 1:-1] + inp[2:, 1:-1] + inp[1:-1, 2:] + inp[:-2, 1:-1] + inp[1:-1, :-2]


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")


def test_temporaries(program_processor_no_transforms):
    program_processor, validate = program_processor_no_transforms
    size = 10
    dtype = float64

    @fendef(offset_provider={"i": IDim, "j": JDim})
    def fencil(inp, out):
        halo_domain = cartesian_domain(
            named_range(IDim, -1, size + 1), named_range(JDim, -1, size + 1)
        )
        domain = cartesian_domain(named_range(IDim, 0, size), named_range(JDim, 0, size))

        tmp = temporary(halo_domain, dtype)

        set_at(
            as_fieldop(lap, halo_domain)(inp),
            halo_domain,
            tmp,
        )

        set_at(
            as_fieldop(lap, domain)(tmp),
            domain,
            out,
        )

    rng = np.random.default_rng()
    inp = gtx.as_field(
        gtx.domain({IDim: (-2, size + 2), JDim: (-2, size + 2)}),
        rng.normal(size=(size + 4, size + 4)),
    )
    out = gtx.as_field([IDim, JDim], np.zeros((size, size)))

    run_processor(fencil, program_processor, inp, out)

    if validate:
        assert np.allclose(out.asnumpy(), lap_ref(lap_ref(inp.asnumpy())))
