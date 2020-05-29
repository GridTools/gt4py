# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

"""Implementation of GTScript: an embedded DSL in Python for stencil computations.

Interface functions to define and compile GTScript definitions and empty symbol
definitions for the keywords of the DSL.
"""

import inspect

from gt4py import definitions as gt_definitions
from gt4py import __gtscript__, __externals__  # noqa
from gt4py.build import BuildContext, LazyStencil

# GTScript builtins
from gt4py.gtscript_impl import (
    _FieldDescriptorMaker,
    _Axis,
    _SequenceDescriptorMaker,
    _ComputationContextManager,
    _set_arg_dtypes,
)

builtins = {
    "I",
    "J",
    "K",
    "IJ",
    "IK",
    "JK",
    "IJK",
    "FORWARD",
    "BACKWARD",
    "PARALLEL",
    "Field",
    "Sequence",
    "externals",
    "computation",
    "interval",
    "__gtscript__",
    "__externals__",
    "__INLINED",
}

__all__ = list(builtins) + ["function", "stencil", "lazy_stencil"]

__externals__ = "Placeholder"
__gtscript__ = "Placeholder"


def function(func):
    """GTScript function."""

    from gt4py.frontend import gtscript_frontend as gt_frontend

    gt_frontend.GTScriptParser.annotate_definition(func)
    return func


# Interface functions
def stencil(
    backend,
    definition=None,
    *,
    build_info=None,
    dtypes=None,
    externals=None,
    name=None,
    rebuild=False,
    **kwargs,
):
    """Generate an implementation of the stencil definition with the specified backend.

    It can be used as a parametrized function decorator or as a regular function.

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        build_info : `dict`, optional
            Dictionary used to store information about the stencil generation.
            (`None` by default).

        dtypes: `dict`[`str`, dtype_definition], optional
            Specify dtypes for string keys in the argument annotations.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`gridtools.StencilObject`
            Properly initialized instance of a dynamically-generated
            subclass of :class:`gt4py.StencilObject`.

    Raises
    -------
        ValueError
            If inconsistent arguments are specified.

    Examples
    --------
        TODO

    """

    from gt4py import loader as gt_loader

    if build_info is not None and not isinstance(build_info, dict):
        raise ValueError(f"Invalid 'build_info' dictionary ('{build_info}')")
    if dtypes is not None and not isinstance(dtypes, dict):
        raise ValueError(f"Invalid 'dtypes' dictionary ('{dtypes}')")
    if externals is not None and not isinstance(externals, dict):
        raise ValueError(f"Invalid 'externals' dictionary ('{externals}')")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"Invalid 'name' string ('{name}')")
    if not isinstance(rebuild, bool):
        raise ValueError(f"Invalid 'rebuild' bool value ('{rebuild}')")

    module = None
    if name:
        name_components = name.split(".")
        name = name_components[-1]
        module = ".".join(name_components[:-1])

    name = name or ""
    module = (
        module or inspect.currentframe().f_back.f_globals["__name__"]
    )  # definition_func.__globals__["__name__"] ??,

    build_options = gt_definitions.BuildOptions(
        name=name, module=module, rebuild=rebuild, backend_opts=kwargs, build_info=build_info
    )

    def _decorator(def_func):
        _set_arg_dtypes(def_func, dtypes or {})
        return gt_loader.gtscript_loader(
            def_func, backend=backend, build_options=build_options, externals=externals or {}
        )

    if definition is None:
        return _decorator
    else:
        return _decorator(definition)


def lazy_stencil(
    backend=None,
    definition=None,
    *,
    build_info=None,
    dtypes=None,
    externals=None,
    name=None,
    rebuild=False,
    eager=False,
    check_syntax=False,
    **kwargs,
):
    """
    Create a stencil object with deferred building and optional up-front syntax checking.

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        build_info : `dict`, optional
            Dictionary used to store information about the stencil generation.
            (`None` by default).

        dtypes: `dict`[`str`, dtype_definition], optional
            Specify dtypes for string keys in the argument annotations.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        eager : `bool`, optional
            If true do not defer stencil building and instead return the fully built raw implementation object.

        check_syntax: `bool`, optional
            If true, build and cache the IR build stage already, which checks stencil definition syntax.

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`gridtools.build.LazyStencil`
            Wrapper arouund an instance of the dynamically-generated subclass of :class:`gt4py.StencilObject`.
            Defers the generation step until the last moment and allows syntax checking independently.
            Also gives access to a more fine grained generate / build process.
    """
    from gt4py import frontend

    def _decorator(func):
        defaults = {
            "backend": backend,
            "frontend": frontend.from_name("gtscript"),
            "dtypes": dtypes,
            "externals": externals or {},
            "build_info": build_info,
            "name": name,
            "rebuild": rebuild,
        }
        defaults.update(kwargs)

        stencil = LazyStencil(BuildContext(func, **defaults))
        if eager:
            stencil = stencil.implementation
        elif check_syntax:
            stencil.check_syntax()
        return stencil

    if definition is None:
        return _decorator
    return _decorator(definition)


# GTScript builtins: domain axes


I = _Axis("I")
"""I axes (parallel)."""

J = _Axis("J")
"""J axes (parallel)."""

K = _Axis("K")
"""K axes (sequential)."""

IJ = (I, J)
"""Tuple of axes I, J."""

IK = (I, K)
"""Tuple of axes I, K."""

JK = (J, K)
"""Tuple of axes J, K."""

IJK = (I, J, K)
"""Tuple of axes I, J, K."""


def mask_from_axes(axes):
    if isinstance(axes, _Axis):
        axes = (axes,)
    axes = list(a.name for a in axes)
    return list(a in axes for a in list(a.name for a in IJK))


# GTScript builtins: iteration orders
FORWARD = +1
"""Forward iteration order."""

BACKWARD = -1
"""Backward iteration order."""

PARALLEL = 0
"""Parallel iteration order."""

# GTScript builtins: variable annotations
Field = _FieldDescriptorMaker()
"""Field descriptor."""

Sequence = _SequenceDescriptorMaker()
"""Sequence descriptor."""


# GTScript builtins: external definitions
def externals(*args):
    """Inlined values of the externals definitions."""
    return args


# GTScript builtins: computation and interval statements


def computation(order):
    """Define the computation."""
    return _ComputationContextManager()


def interval(start, end):
    """Define the interval of computation in the 'K' sequential axis."""
    pass


def __INLINED(compile_if_expression):
    """Evaluate condition at compile time and inline statements from selected branch."""
    pass
