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

"""Global configuration of cartesian test generation and execution (with Hypothesis and pytest)."""

import os
import shutil
from tempfile import mkdtemp

import hypothesis as hyp

from gt4py.cartesian import config as gt_config


def pytest_configure(config):
    # HealthCheck.too_slow causes more trouble than good -- especially in CIs.
    hyp.settings.register_profile(
        "slow", hyp.settings(suppress_health_check=[hyp.HealthCheck.too_slow], deadline=None)
    )
    hyp.settings.load_profile("slow")


# Setup cache folder
if not (pytest_gt_cache_dir := os.environ.get("GT_CACHE_PYTEST_DIR", None)):
    pytest_gt_cache_dir = mkdtemp(
        prefix=".gt_cache_pytest_", dir=gt_config.cache_settings["root_path"]
    )


def pytest_addoption(parser):
    parser.addoption("--keep-gtcache", action="store_true", default=False, dest="keep_gtcache")


def pytest_sessionstart():
    gt_config.cache_settings["dir_name"] = pytest_gt_cache_dir


def pytest_sessionfinish(session):
    if not session.config.option.keep_gtcache:
        shutil.rmtree(pytest_gt_cache_dir, ignore_errors=True)
    else:
        print(f"\nNOTE: gt4py caches were retained at {pytest_gt_cache_dir}")
