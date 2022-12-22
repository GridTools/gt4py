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

"""Global configuration of Hypothesis and pytest for running tests."""


import hypothesis as hyp


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
