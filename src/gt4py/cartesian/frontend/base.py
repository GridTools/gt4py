# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from typing import Any, Type

from gt4py.cartesian import utils as gt_utils
from gt4py.cartesian.definitions import BuildOptions, StencilID
from gt4py.cartesian.gtc import gtir
from gt4py.cartesian.type_hints import AnnotatedStencilFunc, AnyStencilFunc


class Frontend(abc.ABC):
    name: str
    """Frontend name."""

    @classmethod
    @abc.abstractmethod
    def get_stencil_id(
        cls,
        qualified_name: str,
        definition: AnyStencilFunc,
        externals: dict[str, Any],
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
        cls,
        definition: AnyStencilFunc,
        externals: dict[str, Any],
        dtypes: dict[Type, Type],
        options: BuildOptions,
        backend_name: str,
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
        cls, definition: AnyStencilFunc, externals: dict[str, Any]
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


REGISTRY = gt_utils.Registry[Type[Frontend]]()


def from_name(name: str) -> Type[Frontend]:
    """Return frontend by name."""
    frontend_cls = REGISTRY.get(name, None)
    if frontend_cls is None:
        raise NotImplementedError(
            f"Frontend '{name} is not implemented. Options are: {REGISTRY.names}."
        )
    return frontend_cls


def register(frontend_cls: Type[Frontend]) -> Type[Frontend]:
    """Register a new frontend."""
    return REGISTRY.register(frontend_cls.name, frontend_cls)
