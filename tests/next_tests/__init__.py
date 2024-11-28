# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
