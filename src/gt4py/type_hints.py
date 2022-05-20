# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict

from typing_extensions import Protocol


class StencilFunc(Protocol):
    __name__: str
    __module__: str

    def __call__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        ...


class AnnotatedStencilFunc(StencilFunc, Protocol):
    _gtscript_: Dict[str, Any]
