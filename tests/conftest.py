# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""GlobalDecl configuration of test generation and execution (with Hypothesis and pytest)
"""

import os
import shutil

import hypothesis as hyp
import pytest

from gt4py import config as gt_config

from .analysis_setup import (
    AnalysisPass,
    build_iir_pass,
    compute_extents_pass,
    compute_used_symbols_pass,
    demote_locals_pass,
    init_pass,
    merge_blocks_pass,
    normalize_blocks_pass,
)
from .definition_setup import (
    TAssign,
    TComputationBlock,
    TDefinition,
    ij_offset,
    ijk_domain,
    iteration_order,
    non_parallel_iteration_order,
)


# Delete cache folder
shutil.rmtree(
    os.path.join(gt_config.cache_settings["root_path"], gt_config.cache_settings["dir_name"]),
    ignore_errors=True,
)

# Ignore hidden folders and disabled tests
collect_ignore_glob = [".*", "_disabled*"]


def pytest_configure(config):
    # HealthCheck.too_slow causes more trouble than good -- especially in CIs.
    hyp.settings.register_profile(
        "slow", hyp.settings(suppress_health_check=[hyp.HealthCheck.too_slow], deadline=None)
    )
    config.addinivalue_line(
        "markers",
        "requires_cudatoolkit: mark tests that require compilation of CUDA stencils (assume cupy is installed)",
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark tests that require a Nvidia GPU (assume cupy and cudatoolkit are installed)",
    )
    hyp.settings.load_profile("slow")
