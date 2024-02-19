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

import warnings

from gt4py.next import config

from . import definitions


__all__ = ["definitions", "get_processor_id"]


if config.BUILD_CACHE_LIFETIME is config.BuildCacheLifetime.PERSISTENT:
    warnings.warn(
        "You are running GT4Py tests with BUILD_CACHE_LIFETIME set to PERSISTENT!", UserWarning
    )


def get_processor_id(processor):
    if hasattr(processor, "__module__") and hasattr(processor, "__name__"):
        module_path = processor.__module__.split(".")[-1]
        name = processor.__name__
        return f"{module_path}.{name}"
    elif hasattr(processor, "__module__") and hasattr(processor, "__class__"):
        module_path = processor.__module__.split(".")[-1]
        name = processor.__class__.__name__
        return f"{module_path}.{name}"
    return repr(processor)
