# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import functools
import os
import platform
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy

from gt4py.cartesian import utils as gt_utils
from gt4py.cartesian.gtc.definitions import Boundary, CartesianSpace
from gt4py.cartesian.utils.attrib import (
    Any,
    AttributeClassLike,
    Dict as DictOf,
    attribclass,
    attribkwclass,
    attribute,
)


# Dev note: platform.architecture() returns "('64bit', 'ELF')" for example.
_ARCHITECTURE_LITERAL_PRECISION = platform.architecture()[0][:2]
"""Literal precision of the architecture; expected 64 or 32."""

LITERAL_INT_PRECISION = int(
    os.environ.get("GT4PY_LITERAL_INT_PRECISION", default=_ARCHITECTURE_LITERAL_PRECISION)
)
"""Default literal precision used for unspecific `int` types and casts."""

LITERAL_FLOAT_PRECISION = int(
    os.environ.get("GT4PY_LITERAL_FLOAT_PRECISION", default=_ARCHITECTURE_LITERAL_PRECISION)
)
"""Default literal precision used for unspecific `float` types and casts."""


@enum.unique
class AccessKind(enum.IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2
    READ_WRITE = READ | WRITE

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class DomainInfo:
    parallel_axes: Tuple[str, ...]
    sequential_axis: str
    min_sequential_axis_size: int
    ndim: int


@dataclass(frozen=True)
class FieldInfo:
    access: AccessKind
    boundary: Boundary
    axes: Tuple[str, ...]
    data_dims: Tuple[int, ...]
    dtype: numpy.dtype

    def __repr__(self):
        return "FieldInfo(access=AccessKind.{access}, boundary={boundary}, axes={axes}, data_dims={data_dims}, dtype={dtype})".format(
            access=self.access.name,
            boundary=repr(self.boundary),
            axes=repr(self.axes),
            data_dims=repr(self.data_dims),
            dtype=repr(self.dtype),
        )

    @functools.cached_property
    def domain_mask(self):
        return tuple(axis in self.axes for axis in CartesianSpace.names)

    @functools.cached_property
    def domain_ndim(self):
        return len(self.axes)

    @functools.cached_property
    def mask(self):
        return (*self.domain_mask, *((True,) * len(self.data_dims)))

    @functools.cached_property
    def ndim(self):
        return len(self.axes) + len(self.data_dims)


@dataclass(frozen=True)
class ParameterInfo:
    access: Union[Literal[AccessKind.NONE], Literal[AccessKind.READ]]
    dtype: numpy.dtype

    def __repr__(self):
        return "ParameterInfo(access=AccessKind.{access}, dtype={dtype})".format(
            access=self.access.name, dtype=repr(self.dtype)
        )


@attribkwclass
class BuildOptions(AttributeClassLike):
    """Build options."""

    name = attribute(of=str)
    module = attribute(of=str)
    format_source = attribute(of=bool, default=True)
    backend_opts = attribute(of=DictOf[str, Any], factory=dict)
    build_info = attribute(of=dict, optional=True)
    rebuild = attribute(of=bool, default=False)
    raise_if_not_cached = attribute(of=bool, default=False)
    cache_settings = attribute(of=DictOf[str, Any], factory=dict)
    _impl_opts = attribute(of=DictOf[str, Any], factory=dict)
    literal_int_precision = attribute(of=int, default=LITERAL_INT_PRECISION)
    "Literal precision for `int` types and casts. Defaults to architecture precision unless overwritten by the environment variable `GT4PY_LITERAL_INT_PRECISION`."
    literal_float_precision = attribute(of=int, default=LITERAL_FLOAT_PRECISION)
    "Literal precision for `float` types and casts. Defaults to architecture precision unless overwritten by the environment variable `GT4PY_LITERAL_FLOAT_PRECISION`."

    @property
    def qualified_name(self):
        return ".".join([self.module, self.name])

    @property
    def shashed_id(self):
        return gt_utils.shashed_id(
            self.name,
            self.module,
            self.format_source,
            self.literal_int_precision,
            self.literal_float_precision,
            *tuple(sorted(self.backend_opts.items())),
        )


@attribclass(frozen=True)
class StencilID(AttributeClassLike):
    qualified_name = attribute(of=str)
    version = attribute(of=str)

    def __iter__(self):
        return iter([self.qualified_name, self.version])


class GTError(Exception):
    pass


class GTSyntaxError(GTError):
    def __init__(self, message: str, *, frontend: str) -> None:
        super().__init__(message)
        self.frontend = frontend


class GTSpecificationError(GTError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GTSemanticError(GTError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
