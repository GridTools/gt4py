# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared build system functionality."""

from __future__ import annotations

import importlib


def python_module_suffix() -> str:
    return importlib.machinery.EXTENSION_SUFFIXES[0][1:]
