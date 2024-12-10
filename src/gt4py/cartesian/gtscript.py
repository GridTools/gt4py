# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of GTScript: an embedded DSL in Python for stencil computations.

Interface functions to define and compile GTScript definitions and empty symbol
definitions for the keywords of the DSL.
"""

import collections
import inspect
import numbers
import types
from typing import Callable, Dict, Type

import numpy as np

from gt4py.cartesian import definitions as gt_definitions
from gt4py.cartesian.lazy_stencil import LazyStencil


try:
    from gt4py.cartesian.backend.dace_lazy_stencil import DaCeLazyStencil
except ImportError:
    DaCeLazyStencil = LazyStencil  # type: ignore

# GTScript builtins
MATH_BUILTINS = {
    "abs",
    "min",
    "max",
    "mod",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "sqrt",
    "exp",
    "log",
    "log10",
    "gamma",
    "cbrt",
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
    "horizontal",
    "region",
    "__gtscript__",
    "__externals__",
    "__INLINED",
    "compile_assert",
    *MATH_BUILTINS,
}

IGNORE_WHEN_INLINING = {*MATH_BUILTINS, "compile_assert"}

__all__ = [*list(builtins), "function", "stencil", "lazy_stencil"]

__externals__ = "Placeholder"
__gtscript__ = "Placeholder"


_VALID_DATA_TYPES = (
    bool,
    np.bool_,
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
)


def _set_arg_dtypes(definition: Callable[..., None], dtypes: Dict[Type, Type]):
    def _parse_annotation(arg, annotation):
        # This function evaluates the type hint 'annotation' for the stencil argument 'arg'.
        # Note that 'typing.get_type_hints()' cannot be used here since field
        # arguments are annotated using instances (and not subclasses) of 'Field',
        # which is explicitly forbidden by 'get_type_hints()'.
        #
        if isinstance(annotation, _FieldDescriptor) and isinstance(annotation.dtype, str):
            if annotation.dtype in dtypes:
                return _FieldDescriptor(
                    dtypes[annotation.dtype], annotation.axes, annotation.data_dims
                )
            else:
                raise ValueError(f"Missing '{annotation.dtype}' dtype definition for arg '{arg}'")
        elif isinstance(annotation, str):
            if annotation in dtypes:
                return dtypes[annotation]
            else:
                def_globals = getattr(definition, "__globals__", {})
                return _parse_annotation(arg, eval(annotation, def_globals))
        else:
            return annotation

    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    original_annotations = {**annotations}
    for key, value in annotations.items():
        annotations[key] = _parse_annotation(key, value)
    return original_annotations


def function(func):
    """Mark a GTScript function."""
    from gt4py.cartesian.frontend import gtscript_frontend as gt_frontend

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
    format_source=True,
    name=None,
    rebuild=False,
    cache_settings=None,
    raise_if_not_cached=False,
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
            (`None` by default). Possible key-value pairs include:
            - 'symbol_info': (Dict[str, SymbolInfo]) Dictionary of SymbolInfo objects
            - 'parse_time': (float) Frontend run time, e.g., parsing GTScript in seconds
            - 'module_time': (float) Python module generation time in seconds
            - 'codegen_time'" (float) Backend-specific code generation time in seconds
            - 'build_time': (float) Compilation time, i.e., for non-Python backends in seconds
            - 'load_time': (float) Module load time for cached stencils in seconds

        dtypes: `dict`[`str`, dtype_definition], optional
            Specify dtypes for string keys in the argument annotations.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        format_source : `bool`, optional
            Format generated sources when possible (`True` by default).

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        raise_if_not_cached: `bool`, optional
            If this is True, the call will raise an exception if the stencil does not
            exist in the cache, or if the cache is inconsistent. (`False` by default).

        cache_settings: `dict`, optional
            Dictionary to configure cache (directory) settings (see
            ``gt4py.cartesian.config.cache_settings``).
            Possible key-value pairs:
            - `root_path`: (str)
            - `dir_name`: (str)

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

    from gt4py.cartesian import loader as gt_loader

    if build_info is not None and not isinstance(build_info, dict):
        raise ValueError(f"Invalid 'build_info' dictionary ('{build_info}')")
    if dtypes is not None and not isinstance(dtypes, dict):
        raise ValueError(f"Invalid 'dtypes' dictionary ('{dtypes}')")
    if externals is not None and not isinstance(externals, dict):
        raise ValueError(f"Invalid 'externals' dictionary ('{externals}')")
    if not isinstance(format_source, bool):
        raise ValueError(f"Invalid 'format_source' bool value ('{name}')")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"Invalid 'name' string ('{name}')")
    if not isinstance(rebuild, bool):
        raise ValueError(f"Invalid 'rebuild' bool value ('{rebuild}')")
    if not isinstance(raise_if_not_cached, bool):
        raise ValueError(f"Invalid 'raise_if_not_cached' bool value ('{raise_if_not_cached}')")
    if cache_settings is not None and not isinstance(cache_settings, dict):
        raise ValueError(f"Invalid 'cache_settings' dictionary ('{cache_settings}')")

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

    # Setup build_info timings
    if build_info is not None:
        time_keys = ("parse_time", "module_time", "codegen_time", "build_time", "load_time")
        build_info.update({time_key: 0.0 for time_key in time_keys})

    build_options = gt_definitions.BuildOptions(
        name=name,
        module=module,
        format_source=format_source,
        rebuild=rebuild,
        raise_if_not_cached=raise_if_not_cached,
        backend_opts=kwargs,
        build_info=build_info,
        cache_settings=cache_settings or {},
        impl_opts=_impl_opts,
    )

    def _decorator(definition_func):
        if not isinstance(definition_func, types.FunctionType):
            if hasattr(definition_func, "definition_func"):  # StencilObject
                definition_func = definition_func.definition_func
            elif callable(definition_func):  # General callable
                definition_func = definition_func.__call__

        original_annotations = _set_arg_dtypes(definition_func, dtypes or {})
        out = gt_loader.gtscript_loader(
            definition_func,
            backend=backend,
            build_options=build_options,
            externals=externals or {},
            dtypes=dtypes or {},
        )
        definition_func.__annotations__ = original_annotations
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
    format_source=True,
    name=None,
    rebuild=False,
    raise_if_not_cached=False,
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

        format_source : `bool`, optional
            Format generated sources when possible (`True` by default).

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        raise_if_not_cached: `bool`, optional
            If this is True, the call will raise an exception if the stencil does not
            exist in the cache, or if the cache is inconsistent. (`False` by default).

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
    from gt4py.cartesian.stencil_builder import StencilBuilder

    if build_info is not None and not isinstance(build_info, dict):
        raise ValueError(f"Invalid 'build_info' dictionary ('{build_info}')")
    if dtypes is not None and not isinstance(dtypes, dict):
        raise ValueError(f"Invalid 'dtypes' dictionary ('{dtypes}')")
    if externals is not None and not isinstance(externals, dict):
        raise ValueError(f"Invalid 'externals' dictionary ('{externals}')")
    if not isinstance(format_source, bool):
        raise ValueError(f"Invalid 'format_source' bool value ('{name}')")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"Invalid 'name' string ('{name}')")
    if not isinstance(rebuild, bool):
        raise ValueError(f"Invalid 'rebuild' bool value ('{rebuild}')")
    if not isinstance(raise_if_not_cached, bool):
        raise ValueError(f"Invalid 'raise_if_not_cached' bool value ('{raise_if_not_cached}')")

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

    # Setup build_info timings
    if build_info is not None:
        time_keys = ("parse_time", "module_time", "codegen_time", "build_time", "load_time")
        build_info.update({time_key: 0.0 for time_key in time_keys})

    build_options = gt_definitions.BuildOptions(
        name=name,
        module=module,
        format_source=format_source,
        rebuild=rebuild,
        raise_if_not_cached=raise_if_not_cached,
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

        if not build_options.name:
            build_options.name = f"{definition_func.__name__}"
        if backend and "dace" in backend:
            stencil = DaCeLazyStencil(
                StencilBuilder(definition_func, backend=backend, options=build_options)
                .with_externals(externals or {})
                .with_dtypes(dtypes or {})
            )

        else:
            stencil = LazyStencil(
                StencilBuilder(definition_func, backend=backend, options=build_options)
                .with_externals(externals or {})
                .with_dtypes(dtypes or {})
            )
        if eager:
            stencil = stencil.implementation
        elif check_syntax:
            stencil.check_syntax()
        return stencil

    if definition is None:
        return _decorator
    return _decorator(definition)


class AxisIndex:
    def __init__(self, axis: str, index: int, offset: int = 0):
        self.axis = axis
        self.index = index
        self.offset = offset

    def __repr__(self):
        return f"AxisIndex(axis={self.axis}, index={self.index}, offset={self.offset})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __str__(self) -> str:
        return f"{self.axis}[{self.index}] + {self.offset}"

    def __add__(self, offset: int):
        if not isinstance(offset, numbers.Integral):
            raise TypeError("Offset should be an integer type")
        if offset == 0:
            return self
        else:
            return AxisIndex(self.axis, self.index, self.offset + offset)

    def __radd__(self, offset: int):
        return self.__add__(offset)

    def __sub__(self, offset: int):
        return self.__add__(-offset)

    def __rsub__(self, offset: int):
        return self.__radd__(-offset)


class AxisInterval:
    def __init__(self, axis: str, start: int, end: int):
        assert start < end
        self.axis = axis
        self.start = start
        self.end = end

    def __repr__(self):
        return f"AxisInterval(axis={self.axis}, start={self.start}, end={self.end})"

    def __str__(self) -> str:
        return f"{self.axis}[{self.start}:{self.end}]"

    def __len__(self):
        return self.end - self.start


class ShiftedAxis:
    def __init__(self, name: str, shift: int):
        assert name
        self.name = name
        self.shift = shift

    def __repr__(self):
        return f"ShiftedAxis(name={self.name}, shift={self.shift})"

    def __str__(self) -> str:
        return f"{self.name}+{self.shift}"

    def __add__(self, shift):
        if not isinstance(shift, int):
            raise TypeError(f"Can only add type int, got {type(shift)}")
        return ShiftedAxis(self.name, self.shift + shift)

    def __sub__(self, shift):
        if not isinstance(shift, int):
            raise TypeError(f"Can only subtract type int, got {type(shift)}")
        return ShiftedAxis(self.name, self.shift - shift)


# GTScript builtins: domain axes
class Axis:
    def __init__(self, name: str):
        assert name
        self.name = name

    @property
    def __gt_axis_name__(self) -> str:
        return self.name

    def __repr__(self):
        return f"Axis(name={self.name})"

    def __str__(self) -> str:
        return self.name

    def __getitem__(self, interval):
        if isinstance(interval, slice):
            return AxisInterval(self.name, interval.start, interval.stop)
        elif isinstance(interval, int):
            return AxisIndex(self.name, interval)
        else:
            raise TypeError("Unrecognized index type")

    def __add__(self, shift):
        if not isinstance(shift, int):
            raise TypeError(f"Can only add type int, got {type(shift)}")
        return ShiftedAxis(self.name, shift)

    def __sub__(self, shift):
        if not isinstance(shift, int):
            raise TypeError(f"Can only subtract type int, got {type(shift)}")
        return ShiftedAxis(self.name, -shift)


I = Axis("I")  # noqa: E741 [ambiguous name]
"""I axes (parallel)."""

J = Axis("J")
"""J axes (parallel)."""

K = Axis("K")
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
    if isinstance(axes, Axis):
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
    def __init__(self, dtype, axes, data_dims=tuple()):
        if isinstance(dtype, str):
            self.dtype = dtype
        else:
            try:
                dtype = np.dtype(dtype)
                if dtype.shape:
                    assert not data_dims
                    dtype = dtype.base
                    data_dims = dtype.shape
                if dtype not in _VALID_DATA_TYPES:
                    raise ValueError("Invalid data type descriptor")
            except ValueError as ex:
                raise ValueError("Invalid data type descriptor") from ex
            self.dtype = np.dtype(dtype)
        self.axes = axes if isinstance(axes, collections.abc.Collection) else [axes]
        if data_dims:
            if not isinstance(data_dims, collections.abc.Collection):
                self.data_dims = (data_dims,)
            else:
                self.data_dims = tuple(data_dims)
        else:
            self.data_dims = data_dims

    def __descriptor__(self):
        # Ignore, use JIT
        return None

    def __repr__(self):
        args = f"dtype={self.dtype!r}, axes={self.axes!r}, data_dims={self.data_dims!r}"
        return f"_FieldDescriptor({args})"

    def __str__(self) -> str:
        return (
            f"Field<[{', '.join(str(ax) for ax in self.axes)}], ({self.dtype}, {self.data_dims})>"
        )


class _FieldDescriptorMaker:
    @staticmethod
    def _is_axes_spec(spec) -> bool:
        return isinstance(spec, Axis) or (
            isinstance(spec, collections.abc.Collection) and all(isinstance(i, Axis) for i in spec)
        )

    def __getitem__(self, field_spec):
        axes = IJK
        data_dims = ()

        if isinstance(field_spec, str) or not isinstance(field_spec, collections.abc.Collection):
            # Field[dtype] # noqa: ERA001 [commented-out-code]
            dtype = field_spec
        elif _FieldDescriptorMaker._is_axes_spec(field_spec[0]):
            # Field[axes, dtype] # noqa: ERA001 [commented-out-code]
            assert len(field_spec) == 2
            axes, dtype = field_spec
        elif len(field_spec) == 2 and not _FieldDescriptorMaker._is_axes_spec(field_spec[1]):
            # Field[high_dimensional_dtype] # noqa: ERA001 [commented-out-code]
            dtype = field_spec
        else:
            raise ValueError("Invalid field type descriptor")

        if isinstance(dtype, collections.abc.Collection) and not isinstance(dtype, str):
            # high dimensional dtype also includes data axes
            assert len(dtype) == 2
            dtype, data_dims = dtype

        return _FieldDescriptor(dtype, axes, data_dims)


class _GlobalTableDescriptorMaker(_FieldDescriptorMaker):
    def __getitem__(self, field_spec):
        if not isinstance(field_spec, collections.abc.Collection) and not len(field_spec) == 2:
            raise ValueError("GlobalTable is defined by a tuple (type, [axes_size..])")

        dtype, data_dims = field_spec

        return _FieldDescriptor(dtype, [], data_dims)


# GTScript builtins: variable annotations
Field = _FieldDescriptorMaker()
"""Field descriptor."""

GlobalTable = _GlobalTableDescriptorMaker()
"""Data array with no spatial dimension descriptor."""


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


def interval(*args):
    """Define the interval of computation in the 'K' sequential axis."""
    pass


def horizontal(*args):
    """Define a block of code that is restricted to a set of regions in the parallel axes."""
    pass


class _Region:
    def __getitem__(self, *args):
        """Define a region in the parallel axes."""
        pass


# Horizontal regions
region = _Region()


def __INLINED(compile_if_expression):
    """Evaluate condition at compile time and inline statements from selected branch."""
    pass


def compile_assert(expr):
    """Assert that expr evaluates to True at compile-time."""
    pass


# GTScript builtins: math functions
def abs(x):  # noqa: A001 [builtin-variable-shadowing]
    """Return the absolute value of the argument"""
    pass


def min(x, y):  # noqa: A001 [builtin-variable-shadowing]
    """Return the smallest of two or more arguments."""
    pass


def max(x, y):  # noqa: A001 [builtin-variable-shadowing]
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


def sinh(x):
    """Return the hyperbolic sine of x radians"""
    pass


def cosh(x):
    """Return the hyperbolic cosine of x radians."""
    pass


def tanh(x):
    """Return the hyperbolic tangent of x radians."""
    pass


def asinh(x):
    """return the inverse hyperbolic sine of x, in radians."""
    pass


def acosh(x):
    """Return the inverse hyperbolic cosine of x, in radians."""
    pass


def atanh(x):
    """Return the inverse hyperbolic tangent of x, in radians."""
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


def log10(x):
    """Return the base-10 logarithm of x."""
    pass


def gamma(x):
    """Return the gamma function of x."""
    pass


def cbrt(x):
    """Return the cubic root of x."""
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
