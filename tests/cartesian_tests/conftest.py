# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Global configuration of cartesian test generation and execution (with Hypothesis and pytest)."""

import os
import shutil
from tempfile import mkdtemp

import hypothesis as hyp

from gt4py.cartesian import config as gt_config


def pytest_configure(config):
    # HealthCheck.too_slow causes more trouble than good -- especially in CIs.
    hyp.settings.register_profile(
        "lenient",
        hyp.settings(
            suppress_health_check=[hyp.HealthCheck.too_slow, hyp.HealthCheck.data_too_large],
            deadline=None,
        ),
    )
    hyp.settings.load_profile("lenient")


# Setup cache folder
if not (pytest_gt_cache_dir := os.environ.get("GT_CACHE_PYTEST_DIR", None)):
    pytest_gt_cache_dir = mkdtemp(
        prefix=".gt_cache_pytest_", dir=gt_config.cache_settings["root_path"]
    )


def pytest_addoption(parser):
    parser.addoption("--keep-gtcache", action="store_true", default=False, dest="keep_gtcache")


def pytest_sessionstart(session):
    gt_config.cache_settings["dir_name"] = pytest_gt_cache_dir
    if session.config.option.keep_gtcache:
        print(f"\nNOTE: gt4py caches will be retained at {pytest_gt_cache_dir}")


def pytest_sessionfinish(session):
    if not session.config.option.keep_gtcache:
        shutil.rmtree(pytest_gt_cache_dir, ignore_errors=True)
