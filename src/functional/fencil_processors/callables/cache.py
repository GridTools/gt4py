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


import enum
import hashlib
import pathlib
import tempfile


class Strategy(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


_session_cache_dir = tempfile.TemporaryDirectory(prefix="gt4py_session_")

_session_cache_dir_path = pathlib.Path(_session_cache_dir.name)
_persistent_cache_dir_path = (
    pathlib.Path(tempfile.tempdir) / "gt4py_cache" if tempfile.tempdir else _session_cache_dir_path
)


def _cache_folder_name(module_name: str, module_src: str) -> str:
    hashed = hashlib.sha256(module_src.encode(encoding="utf-8"))
    hashed_str = hashed.hexdigest()
    return module_name + "_" + hashed_str


def get_cache_folder(module_name: str, module_src: str, strategy: Strategy) -> pathlib.Path:
    folder_name = _cache_folder_name(module_name, module_src)

    base_path = {
        Strategy.SESSION: _session_cache_dir_path,
        Strategy.PERSISTENT: _persistent_cache_dir_path,
    }[strategy]
    base_path.mkdir(exist_ok=True)

    complete_path = base_path / folder_name
    complete_path.mkdir(exist_ok=True)

    return complete_path
