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


def _xdist_worker_id(session) -> str:
    """Return this process's pytest-xdist worker id (``gw0``, ``gw1``, ...),
    or ``"master"`` when running without xdist (``-n 0``) or in the
    controller. Mirrors the ``worker_id`` fixture without needing a fixture
    (we need it at session-start, before fixtures run)."""
    # xdist sets this env var in each worker subprocess; absent in serial runs.
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def pytest_sessionstart(session):
    global pytest_gt_cache_dir

    # Per-worker cache isolation (parallel determinism runs).
    #
    # gt4py.cartesian compiles each stencil into a dace build folder named by
    # the SDFG hash. Byte-identical stencils -> identical folder name. Under
    # pytest-xdist, two workers compiling the same stencil would target the
    # SAME build folder; on a CMake configure retry dace does
    # `shutil.rmtree(build_folder); makedirs; cmake` (dace/codegen/compiler.py),
    # so one worker can delete the folder out from under another worker's
    # running compiler, wedging it until the CI time limit. Giving each worker
    # its own cache subdirectory keeps identical stencils in distinct trees, so
    # the race cannot occur and the determinism suite can run in parallel.
    #
    # The determinism comparator strips this `gw<N>` / `master` segment from
    # program ids (see scripts/dace_deterministic_codegen.py:_strip_worker_segment),
    # so isolating per worker here does not affect the run1-vs-run2 comparison.
    # Serial runs (`-n 0`) get the `master` segment, which the comparator
    # strips identically — so serial and parallel snapshots compare cleanly.
    worker_id = _xdist_worker_id(session)
    if worker_id != "master":
        pytest_gt_cache_dir = os.path.join(pytest_gt_cache_dir, worker_id)
        os.makedirs(pytest_gt_cache_dir, exist_ok=True)

    gt_config.cache_settings["dir_name"] = pytest_gt_cache_dir
    if session.config.option.keep_gtcache:
        print(f"\nNOTE: gt4py caches will be retained at {pytest_gt_cache_dir}\n")


def pytest_sessionfinish(session):
    if not session.config.option.keep_gtcache:
        shutil.rmtree(pytest_gt_cache_dir, ignore_errors=True)
