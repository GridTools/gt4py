# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import functools
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
    literal_precision = attribute(of=int, default=64)
    "Specify the literal precision for automatic casts. Defaults to 64-bit"

    @property
    def qualified_name(self):
        name = ".".join([self.module, self.name])
        return name

    @property
    def shashed_id(self):
        result = gt_utils.shashed_id(
            self.name, self.module, self.format_source, *tuple(sorted(self.backend_opts.items()))
        )

        return result


@attribclass(frozen=True)
class StencilID(AttributeClassLike):
    qualified_name = attribute(of=str)
    version = attribute(of=str)

    def __iter__(self):
        return iter([self.qualified_name, self.version])


class GTError(Exception):
    pass


class GTSyntaxError(GTError):
    def __init__(self, message, *, frontend):
        super().__init__(message)
        self.frontend = frontend


class GTSpecificationError(GTError):
    def __init__(self, message):
        super().__init__(message)


class GTSemanticError(GTError):
    def __init__(self, message):
        super().__init__(message)
