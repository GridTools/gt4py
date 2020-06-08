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
import functools
import inspect
import types
import numbers

import numpy as np

import gt4py
from gt4py import definitions as gt_definitions
from gt4py import utils as gt_utils


# GTScript builtins
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

__all__ = list(builtins) + ["function", "stencil"]


_VALID_DATA_TYPES = (bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64)


def _set_arg_dtypes(definition, dtypes):
    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    for arg, value in annotations.items():
        if isinstance(value, _FieldDescriptor) and isinstance(value.dtype, str):
            if value.dtype in dtypes:
                annotations[arg].dtype = _DType.normalize_value(dtypes[value.dtype])
            else:
                raise ValueError(f"Missing '{value.dtype}' dtype definition for arg '{arg}'")
        elif isinstance(value, str):
            if value in dtypes:
                annotations[arg] = dtypes[value]
            else:
                raise ValueError(f"Missing '{value}' dtype definition for arg '{arg}'")

    return definition


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

        definition : ``None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        build_info : `dict`, optional
            Dictionary used to store information about the stencil generation.
            (`None` by default).

        dtypes: `dict`['str`, dtype_definition], optional
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


# GTScript builtins: domain axes
class _Axis:
    def __init__(self, name: str):
        assert name in "IJK"
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


class _FieldAnnotationPropertyMaker:
    def __init__(self, property: type):
        self.property = property

    def __getitem__(self, value):
        return self.property(value)


import abc


class _FieldAnnotationProperty(abc.ABC):
    def __init__(self, value):
        self._value = self.normalize_value(value)

    @staticmethod
    @abc.abstractmethod
    def normalize_value(value):
        raise NotImplementedError

    @property
    def value(self):
        return self._value


class _DType(_FieldAnnotationProperty):
    name = "dtype"

    @staticmethod
    def normalize_value(value):
        try:
            dtype = gt4py.ir.nodes.DataType.from_dtype(np.dtype(value))
        except:
            if isinstance(value, str):
                return value
            else:
                raise TypeError(f"Invalid data type descriptor: dtype '{value}' not understood")
        if dtype is gt4py.ir.nodes.DataType.INVALID:
            raise ValueError(f"Invalid data type descriptor: dtype '{value}' not supported")
        return dtype


class _LayoutMap(_FieldAnnotationProperty):
    name = "layout_map"

    @staticmethod
    def normalize_value(value):
        if (
            not gt_utils.is_iterable_of(value, int)
            or not len(value) == 3
            or not 0 in value
            or not 1 in value
            or not 2 in value
        ):
            raise TypeError("LayoutMap must be a permutation of (0, 1, 2).")
        return tuple(value)


LayoutMap = _FieldAnnotationPropertyMaker(_LayoutMap)


class _Alignment(_FieldAnnotationProperty):
    name = "alignment"

    @staticmethod
    def normalize_value(value):
        if not isinstance(value, numbers.Integral) or not value > 0:
            raise ValueError("Alignment must be a positive integer")
        return int(value)


Alignment = _FieldAnnotationPropertyMaker(_Alignment)


class _DefaultParameters(_FieldAnnotationProperty):
    name = "default_parameters"

    @staticmethod
    def normalize_value(value):
        if not isinstance(value, str) or value not in gt4py.storage.default_parameters.REGISTRY:
            raise ValueError(
                f"DefaultParameters must be one of {list(gt4py.storage.default_parameters.REGISTRY.keys())}"
            )
        return gt4py.storage.default_parameters.REGISTRY[value]


DefaultParameters = _FieldAnnotationPropertyMaker(_DefaultParameters)


class _Axes(_FieldAnnotationProperty):
    name = "axes"

    @staticmethod
    def normalize_value(value):
        if isinstance(value, _Axis):
            value = value.name
        elif gt_utils.is_iterable_of(value, _Axis):
            value = "".join(v.name for v in value)
        elif not isinstance(value, str):
            raise TypeError("Invalid Axes type")
        if (
            not all(c in "IJK" for c in value)
            or len(set(value)) != len(value)
            or sorted(value) != list(value)
        ):
            raise ValueError('Axes must be a sub-sequence of "IJK"')
        return tuple(_Axis(c) for c in value)


class _FieldDescriptor:
    def __init__(self, *properties):

        # self._dtype = None
        # self._axes = None
        # self._layout_map = None
        # self._alignment = None
        # self._default_parameters = None
        # self._backend_defaults = None

        defined_properties = set()
        for prop in properties:
            if not isinstance(prop, _FieldAnnotationProperty):
                raise TypeError(f"Field property of type '{type(prop)}' not understood.")
            if id(type(prop)) in defined_properties:
                raise ValueError(
                    f"Repeated specification of {type(prop).__name__} field property."
                )
            defined_properties.add(id(type(prop)))

        dtype = list(prop for prop in properties if isinstance(prop, _DType))
        if len(dtype) == 1:
            self.dtype = dtype[0].value
        else:
            assert len(dtype) == 0
            raise ValueError("Missing dtype field property.")

        axes = list(prop for prop in properties if isinstance(prop, _Axes))
        if len(axes) == 1:
            self.axes = axes[0].value
        else:
            assert len(axes) == 0
            raise ValueError("Missing axes field property.")

        self._backend_defaults = len(properties) <= 2

        if not self._backend_defaults:
            self.alignment = 1
            self.layout_map = (0, 1, 2)
            default_parameters = list(
                prop for prop in properties if isinstance(prop, _DefaultParameters)
            )
            if len(default_parameters) == 1:
                default_parameters = default_parameters[0].value
                if default_parameters.alignment is not None:
                    self.alignment = default_parameters.alignment
                if default_parameters.layout_map is not None:
                    self.layout_map = default_parameters.layout_map
            else:
                assert len(default_parameters) == 0

            alignment = list(prop for prop in properties if isinstance(prop, _Alignment))
            if len(alignment) == 1:
                self.alignment = alignment[0].value
            else:
                assert len(alignment) == 0

            layout_map = list(prop for prop in properties if isinstance(prop, _LayoutMap))
            if len(layout_map) == 1:
                layout_map = layout_map[0].value
            else:
                assert len(layout_map) == 0
            axes = [a.name for a in self._axes]
            mask = [c in axes for c in "IJK"]
            masked_layout = tuple(l for m, l in zip(mask, layout_map) if m)
            self.layout_map = tuple(int(l) for l in np.argsort(np.argsort(masked_layout)))

    def __repr__(self):
        return f"_FieldDescriptor(dtype={repr(self.dtype)}, axes={repr(self.axes)})"

    def __str__(self):
        return f"Field<{str(self.dtype)}, [{', '.join(str(ax) for ax in self.axes)}]>"


class _FieldDescriptorMaker:
    def __getitem__(self, properties):
        if isinstance(properties, collections.abc.Collection) and not isinstance(properties, str):
            properties = list(properties)
        else:
            properties = [properties]
        properties[0] = _FieldAnnotationPropertyMaker(_DType)[properties[0]]

        if len(properties) > 1:
            if not isinstance(properties[1], _FieldAnnotationProperty):
                properties[1] = _FieldAnnotationPropertyMaker(_Axes)[properties[1]]

        defined_properties = set()
        for prop in properties:
            defined_properties.add(id(type(prop)))
        if id(_Axes) not in defined_properties:
            properties.append(_FieldAnnotationPropertyMaker(_Axes)[IJK])
        return _FieldDescriptor(*properties)


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


def __INLINED(compile_if_expression):
    """Evaluate condition at compile time and inline statements from selected branch."""
    pass
