# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import abc
from typing import Any, Dict, Union

from gt4py import utils as gt_utils
from gt4py.definitions import BuildOptions, StencilID
from gt4py.ir import StencilDefinition
from gt4py.type_hints import AnnotatedStencilFunc, StencilFunc


REGISTRY = gt_utils.Registry()
AnyStencilFunc = Union[StencilFunc, AnnotatedStencilFunc]


def from_name(name: str):
    return REGISTRY.get(name, None)


def register(frontend_cls):
    assert issubclass(frontend_cls, Frontend) and frontend_cls.name is not None
    return REGISTRY.register(frontend_cls.name, frontend_cls)


class Frontend(abc.ABC):
    name = None

    @classmethod
    @abc.abstractmethod
    def get_stencil_id(
        cls,
        qualified_name: str,
        definition: AnyStencilFunc,
        externals: Dict[str, Any],
        options_id: str,
    ) -> StencilID:
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls, definition: AnyStencilFunc, externals: Dict[str, Any], options: BuildOptions
    ) -> StencilDefinition:
        pass

    @classmethod
    @abc.abstractmethod
    def prepare_stencil_definition(
        cls, definition: AnyStencilFunc, externals: Dict[str, Any]
    ) -> AnnotatedStencilFunc:
        pass
