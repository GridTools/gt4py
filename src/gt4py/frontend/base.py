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

import abc
from typing import Any, Dict, Type, Union

from gt4py import utils as gt_utils
from gt4py.definitions import BuildOptions, StencilID
from gt4py.type_hints import AnnotatedStencilFunc, StencilFunc
from gtc import gtir


REGISTRY = gt_utils.Registry()
AnyStencilFunc = Union[StencilFunc, AnnotatedStencilFunc]


def from_name(name: str) -> Type["Frontend"]:
    """Return frontend by name."""
    return REGISTRY.get(name, None)


def register(frontend_cls: Type["Frontend"]) -> None:
    """Register a new frontend."""
    return REGISTRY.register(frontend_cls.name, frontend_cls)


class Frontend(abc.ABC):

    name: str
    """Frontend name."""

    @classmethod
    @abc.abstractmethod
    def get_stencil_id(
        cls,
        qualified_name: str,
        definition: AnyStencilFunc,
        externals: Dict[str, Any],
        options_id: str,
    ) -> StencilID:
        """
        Create a StencilID object that contains a unique hash for the stencil.

        Notes
        -----
        This method seems to no longer be used through StencilBuilder.

        Returns
        -------
        StencilID:
            An object that contains the qualified name and unique hash.

        Raises
        ------
        GTSyntaxError
            If there is a parsing error.

        TypeError
            If there is a error resolving external types.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def generate(
        cls, definition: AnyStencilFunc, externals: Dict[str, Any], options: BuildOptions
    ) -> gtir.Stencil:
        """
        Generate a StencilDefinition from a stencil Python function.

        Raises
        ------
        GTScriptSyntaxError
            If there is a parsing error.

        GTScriptDefinitionError
            If types are misused in the definition.

        TypeError
            If there is an error in resolving external types.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def prepare_stencil_definition(
        cls, definition: AnyStencilFunc, externals: Dict[str, Any]
    ) -> AnnotatedStencilFunc:
        """
        Annotate the stencil function if not already done so.

        Raises
        ------
        GTSyntaxError
            If there is a parsing error.

        TypeError
            If there is a error resolving external types.
        """
        pass
