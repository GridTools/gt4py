# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import pathlib
import sys
from types import ModuleType


def import_from_path(
    module_file: pathlib.Path, *, add_to_sys_modules: bool = True, sys_modules_prefix: str = ""
) -> ModuleType:
    """
    Load dynamically a python module from a given file path.

    Args:
        module_file: The path to the python file to load as a module.
        add_to_sys_modules: Whether to add the loaded module to 'sys.modules'.
        sys_modules_prefix: If 'add_to_sys_modules' is `True`, the prefix to use
            for the module name in 'sys.modules'. If empty, the module name will be
            the file name without extension.

    Returns:
        The loaded module.
    """
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

    if add_to_sys_modules:
        if not sys_modules_prefix.endswith("."):
            sys_modules_prefix += "."
        sys.modules[sys_modules_prefix + module_name] = module
    elif sys_modules_prefix:
        raise ValueError("Cannot use 'sys_modules_prefix' if 'add_to_sys_modules' is False")

    return module
