# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import pytest
import itertools
import numpy as np

import hypothesis as hyp
import hypothesis.strategies as hyp_st

from gt4py import backend as gt_backend
from gt4py import storage as gt_store
from .iir_stencil_definitions import REGISTRY as iir_registry
from .utils import generate_test_module

from .utils import id_version


@pytest.mark.parametrize(
    ["name", "backend"],
    itertools.product(
        iir_registry.names,
        [
            gt_backend.from_name(name)
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).storage_info["device"] == "cpu"
        ],
    ),
)
def test_generation_cpu(name, backend, *, id_version):
    generate_test_module(name, backend, id_version=id_version)


@pytest.mark.requires_cudatoolkit
@pytest.mark.parametrize(
    ["name", "backend"],
    itertools.product(
        iir_registry.names,
        [
            gt_backend.from_name(name)
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).storage_info["device"] == "gpu"
        ],
    ),
)
def test_generation_gpu(name, backend, *, id_version):
    generate_test_module(name, backend, id_version=id_version)
