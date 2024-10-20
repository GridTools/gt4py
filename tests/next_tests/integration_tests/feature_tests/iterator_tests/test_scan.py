# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import cartesian_domain, deref, named_range, scan, shift
from gt4py.next.iterator.runtime import fundef, offset

from next_tests.integration_tests.cases import IDim, KDim
from next_tests.unit_tests.conftest import program_processor, run_processor


@pytest.mark.uses_index_fields
def test_scan_in_stencil(program_processor):
    # FIXME[#1582](tehrengruber): Remove test after scan is reworked.
    pytest.skip("Scan inside of stencil is not supported in GTIR.")
    program_processor, validate = program_processor

    isize = 1
    ksize = 3
    Koff = offset("Koff")
    inp = gtx.as_field(
        [IDim, KDim],
        np.copy(np.broadcast_to(np.arange(0, ksize, dtype=np.float64), (isize, ksize))),
    )
    out = gtx.as_field([IDim, KDim], np.zeros((isize, ksize)))

    reference = np.zeros((isize, ksize - 1))
    reference[:, 0] = inp.ndarray[:, 0] + inp.ndarray[:, 1]
    for k in range(1, ksize - 1):
        reference[:, k] = reference[:, k - 1] + inp.ndarray[:, k] + inp.ndarray[:, k + 1]

    @fundef
    def sum(state, k, kp):
        return state + deref(k) + deref(kp)

    @fundef
    def wrapped(inp):
        return scan(sum, True, 0.0)(inp, shift(Koff, 1)(inp))

    run_processor(
        wrapped[cartesian_domain(named_range(IDim, 0, isize), named_range(KDim, 0, ksize - 1))],
        program_processor,
        inp,
        out=out,
        offset_provider={"Koff": KDim},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(out[:, :-1].asnumpy(), reference)
