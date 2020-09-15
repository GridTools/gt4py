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

import collections
import inspect
import types

import numpy as np

from gt4py import definitions as gt_definitions
from gt4py import utils as gt_utils
from gt4py.lazy_stencil import LazyStencil
from gt4py.stencil_builder import StencilBuilder


# GTScript builtins
MATH_BUILTINS = {
    "abs",
    "min",
    "max",
    "mod",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sqrt",
    "exp",
    "log",
    "isfinite",
    "isinf",
    "isnan",
    "floor",
    "ceil",
    "trunc",
}

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
    "parallel",
    "region",
    "__gtscript__",
    "__externals__",
    "__INLINED",
    *MATH_BUILTINS,
}

__all__ = list(builtins) + ["function", "stencil", "lazy_stencil"]

__externals__ = "Placeholder"
__gtscript__ = "Placeholder"


_VALID_DATA_TYPES = (bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64)


def _set_arg_dtypes(definition, dtypes):
    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    original_annotations = {**annotations}
    for arg, value in annotations.items():
        if isinstance(value, _FieldDescriptor) and isinstance(value.dtype, str):
            if value.dtype in dtypes:
                annotations[arg] = _FieldDescriptor(dtypes[value.dtype], value.axes)
            else:
                raise ValueError(f"Missing '{value.dtype}' dtype definition for arg '{arg}'")
        elif isinstance(value, str):
            if value in dtypes:
                annotations[arg] = dtypes[value]
            else:
                raise ValueError(f"Missing '{value}' dtype definition for arg '{arg}'")

    return definition, original_annotations


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

    # Move hidden "_option" keys to _impl_opts
    _impl_opts = {}
    for key, value in kwargs.items():
        if key.startswith("_"):
            _impl_opts[key] = value
    for key in _impl_opts:
        kwargs.pop(key)

    build_options = gt_definitions.BuildOptions(
        name=name,
        module=module,
        rebuild=rebuild,
        backend_opts=kwargs,
        build_info=build_info,
        impl_opts=_impl_opts,
    )

    def _decorator(definition_func):
        if not isinstance(definition_func, types.FunctionType):
            if hasattr(definition_func, "definition_func"):  # StencilObject
                definition_func = definition_func.definition_func
            elif callable(definition_func):  # General callable
                definition_func = definition_func.__call__

        _, original_annotations = _set_arg_dtypes(definition_func, dtypes or {})
        out = gt_loader.gtscript_loader(
            definition_func,
            backend=backend,
            build_options=build_options,
            externals=externals or {},
        )
        setattr(definition_func, "__annotations__", original_annotations)
        return out

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
    check_syntax=True,
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

        check_syntax: `bool`, default=True, optional
            If true, build and cache the IR build stage already, which checks stencil definition syntax.

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`gridtools.build.LazyStencil`
            Wrapper around an instance of the dynamically-generated subclass of :class:`gt4py.StencilObject`.
            Defers the generation step until the last moment and allows syntax checking independently.
            Also gives access to a more fine grained generate / build process.
    """
    from gt4py import frontend

    def _decorator(func):
        _set_arg_dtypes(func, dtypes or {})
        options = gt_definitions.BuildOptions(
            **{
                **StencilBuilder.default_options_dict(func),
                **StencilBuilder.name_to_options_args(name),
                "rebuild": rebuild,
                "build_info": build_info,
                **StencilBuilder.nest_impl_options(kwargs),
            }
        )
        stencil = LazyStencil(
            StencilBuilder(func, backend=backend, options=options).with_externals(externals or {})
        )
        if eager:
            stencil = stencil.implementation
        elif check_syntax:
            stencil.check_syntax()
        return stencil

    if definition is None:
        return _decorator
    return _decorator(definition)


# GTScript builtins: domain axes
class _Axis:
    def __init__(self, name: str):
        assert name
        self.name = name

    def __repr__(self):
        return f"_Axis(name={self.name})"

    def __str__(self):
        return self.name


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


class _FieldDescriptor:
    def __init__(self, dtype, axes):
        if isinstance(dtype, str):
            self.dtype = dtype
        else:
            if dtype not in _VALID_DATA_TYPES:
                raise ValueError("Invalid data type descriptor")
            self.dtype = np.dtype(dtype)
        self.axes = axes if isinstance(axes, collections.abc.Collection) else [axes]

    def __repr__(self):
        return f"_FieldDescriptor(dtype={repr(self.dtype)}, axes={repr(self.axes)})"

    def __str__(self):
        return f"Field<{str(self.dtype)}, [{', '.join(str(ax) for ax in self.axes)}]>"


class _FieldDescriptorMaker:
    def __getitem__(self, dtype_and_axes):
        if isinstance(dtype_and_axes, collections.abc.Collection) and not isinstance(
            dtype_and_axes, str
        ):
            dtype, axes = dtype_and_axes
        else:
            dtype, axes = [dtype_and_axes, IJK]
        return _FieldDescriptor(dtype, axes)


# GTScript builtins: variable annotations
Field = _FieldDescriptorMaker()
"""Field descriptor."""


class _SequenceDescriptor:
    def __init__(self, dtype, length):
        self.dtype = dtype
        self.length = length


class _SequenceDescriptorMaker:
    def __getitem__(self, dtype, length=None):
        return dtype, length


Sequence = _SequenceDescriptorMaker()
"""Sequence descriptor."""


# GTScript builtins: external definitions
def externals(*args):
    """Inlined values of the externals definitions."""
    return args


# GTScript builtins: computation and interval statements
class _ComputationContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def computation(order):
    """Define the computation."""
    return _ComputationContextManager()


def interval(start, end):
    """Define the interval of computation in the 'K' sequential axis."""
    pass


def parallel(*args):
    """Define restricted computations in the parallel 'I-J' plane."""
    pass


def __INLINED(compile_if_expression):
    """Evaluate condition at compile time and inline statements from selected branch."""
    pass


# GTScript builtins: math functions
def abs(x):
    """Return the absolute value of the argument"""
    pass


def min(x, y):
    """Return the smallest of two or more arguments."""
    pass


def max(x, y):
    """Return the largest of two or more arguments."""
    pass


def mod(x, y):
    """returns the first argument modulo the second one"""
    pass


def sin(x):
    """Return the sine of x radians"""
    pass


def cos(x):
    """Return the cosine of x radians."""
    pass


def tan(x):
    """Return the tangent of x radians."""
    pass


def asin(x):
    """return the arc sine of x, in radians."""
    pass


def acos(x):
    """Return the arc cosine of x, in radians."""
    pass


def atan(x):
    """Return the arc tangent of x, in radians."""
    pass


def sqrt(x):
    """Return the square root of x."""
    pass


def exp(x):
    """Return e raised to the power x, where e is the base of natural logarithms."""
    pass


def log(x):
    """Return the natural logarithm of x (to base e)."""
    pass


def isfinite(x):
    """Return True if x is neither an infinity nor a NaN, and False otherwise. (Note that 0.0 is considered finite.)"""
    pass


def isinf(x):
    """Return True if x is a positive or negative infinity, and False otherwise."""
    pass


def isnan(x):
    """Return True if x is a NaN (not a number), and False otherwise."""
    pass


def floor(x):
    """Return the floor of x, the largest integer less than or equal to x."""
    pass


def ceil(x):
    """Return the ceiling of x, the smallest integer greater than or equal to x."""
    pass


def trunc(x):
    """Return the Real value x truncated to an Integral (usually an integer)"""
    pass


class _Region:
    def __getitem__(self, *intervals):
        iaxis, jaxis = intervals[0]
        return ((iaxis.start, iaxis.stop), (jaxis.start, jaxis.stop))


# Horizontal regions
region = _Region()
