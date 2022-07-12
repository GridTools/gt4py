# GT4Py Project - GridTools Framework
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


import importlib
import importlib.util
import inspect
import pathlib
from typing import Callable, Dict


def import_callables(module_file: pathlib.Path) -> Dict[str, Callable]:
    module_name = module_file.name.split(".")[0]

    error_msg = f"Could not load module named {module_name} from {module_file}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if not spec or not spec.loader:
        raise ModuleNotFoundError(error_msg)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        raise ModuleNotFoundError(error_msg)

    members = inspect.getmembers(module)
    functions = filter(lambda obj: inspect.isfunction(obj[1]) or inspect.isbuiltin(obj[1]), members)
    return dict(functions)
