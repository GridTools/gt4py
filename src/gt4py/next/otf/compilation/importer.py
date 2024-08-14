# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.util
import pathlib
from types import ModuleType


def import_from_path(module_file: pathlib.Path) -> ModuleType:
    """Import all function objects from a Python module and return a mapping {function_name: object}."""
    module_name = module_file.name.split(".")[0]

    error_msg = f"Could not load module named {module_name} from {module_file}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if not spec or not spec.loader:
        raise ModuleNotFoundError(error_msg)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as ex:
        raise ModuleNotFoundError(error_msg) from ex

    return module
