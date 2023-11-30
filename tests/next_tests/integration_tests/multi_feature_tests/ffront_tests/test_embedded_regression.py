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

import pytest

from gt4py import next as gtx
from gt4py.next import errors

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IField, cartesian_case  # noqa: F401 # fixtures
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401 # fixtures
    fieldview_backend,
)


def test_default_backend_is_respected(cartesian_case):  # noqa: F811 # fixtures
    """Test that manually calling the field operator without setting the backend raises an error."""

    # Important not to set the backend here!
    @gtx.field_operator
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()

    with pytest.raises(ValueError, match="No backend selected!"):
        # Calling this should fail if the default backend is respected
        # due to `fieldview_backend` fixture (dependency of `cartesian_case`)
        # setting the default backend to something invalid.
        _ = copy(a, out=a, offset_provider={})


def test_missing_arg(cartesian_case):  # noqa: F811 # fixtures
    """Test that calling a field_operator without required args raises an error."""

    @gtx.field_operator(backend=cartesian_case.backend)
    def copy(a: IField) -> IField:
        return a

    a = cases.allocate(cartesian_case, copy, "a")()

    with pytest.raises(errors.MissingArgumentError, match="'out'"):
        _ = copy(a, offset_provider={})

    with pytest.raises(errors.MissingArgumentError, match="'offset_provider'"):
        _ = copy(a, out=a)
