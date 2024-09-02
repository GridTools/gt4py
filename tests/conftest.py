# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Global configuration of pytest for collecting and running tests."""

import pytest


# Ignore hidden folders and disabled tests
collect_ignore_glob = [".*", "_disabled*"]


def pytest_addoption(parser):
    group = parser.getgroup("This project")
    group.addoption(
        "--ignore-no-tests-collected",
        action="store_true",
        default=False,
        help='Suppress the "no tests were collected" exit code.',
    )


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--ignore-no-tests-collected"):
        if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = pytest.ExitCode.OK
