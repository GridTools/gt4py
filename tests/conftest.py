# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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
from tempfile import mkdtemp

import hypothesis as hyp
import pytest

from gt4py import config as gt_config

from .definition_setup import (
    TAssign,
    TComputationBlock,
    TDefinition,
    ij_offset,
    ijk_domain,
    iteration_order,
    non_parallel_iteration_order,
)


# Setup cache folder
if not (pytest_gt_cache_dir := os.environ.get("GT_CACHE_PYTEST_DIR", None)):
    pytest_gt_cache_dir = mkdtemp(
        prefix=".gt_cache_pytest_", dir=gt_config.cache_settings["root_path"]
    )


def pytest_sessionstart():
    gt_config.cache_settings["dir_name"] = pytest_gt_cache_dir


def pytest_sessionfinish(session):
    if not session.config.option.keep_gtcache:
        shutil.rmtree(pytest_gt_cache_dir, ignore_errors=True)
    else:
        print(f"\nNOTE: gt4py caches were retained at {pytest_gt_cache_dir}")


# Ignore hidden folders and disabled tests
collect_ignore_glob = [".*", "_disabled*"]


def pytest_configure(config):
    # HealthCheck.too_slow causes more trouble than good -- especially in CIs.
    hyp.settings.register_profile(
        "slow", hyp.settings(suppress_health_check=[hyp.HealthCheck.too_slow], deadline=None)
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark tests that require a Nvidia GPU (assume cupy and cudatoolkit are installed)",
    )
    config.addinivalue_line(
        "markers",
        "requires_dace: mark tests that require dace in the python environment",
    )
    hyp.settings.load_profile("slow")


def pytest_addoption(parser):
    parser.addoption("--keep-gtcache", action="store_true", default=False, dest="keep_gtcache")
