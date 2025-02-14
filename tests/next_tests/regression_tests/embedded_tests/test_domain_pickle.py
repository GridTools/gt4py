# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pickle

from gt4py.next import common

I = common.Dimension("I")
J = common.Dimension("J")


def test_domain_pickle_after_slice():
    domain = common.domain(((I, (2, 4)), (J, (3, 5))))
    # use slice_at to populate cached property
    domain.slice_at[2:5, 5:7]

    pickle.dumps(domain)
