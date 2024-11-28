# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict

from typing_extensions import Protocol


class StencilFunc(Protocol):
    __name__: str
    __module__: str

    def __call__(self, *args: Any, **kwargs: Dict[str, Any]) -> None: ...


class AnnotatedStencilFunc(StencilFunc, Protocol):
    _gtscript_: Dict[str, Any]
