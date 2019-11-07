# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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
import collections
import enum
import numbers
import operator
import sys
import time
import warnings


from gt4py import utils as gt_utils
import gt4py as gt
from gt4py.utils.attrib import (
    attribute,
    attribclass,
    attribkwclass,
    Any,
    Dict as DictOf,
    List as ListOf,
    Tuple as TupleOf,
)


class CartesianSpace:
    @enum.unique
    class Axis(enum.Enum):
        I = 0
        J = 1
        K = 2

        def __str__(self):
            return self.name

    names = [ax.name for ax in Axis]
    ndims = len(names)


class NumericTuple(tuple):
    """N-dimensional like vector implemented as a subclass of the tuple builtin."""

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(isinstance(d, numbers.Number) for d in value)
        assert ndims[0] <= len(value) <= ndims[1]

    @classmethod
    def is_valid(cls, value, *, ndims=(1, CartesianSpace.ndims)):
        if isinstance(ndims, int):
            ndims = tuple([ndims] * 2)
        elif not isinstance(ndims, collections.abc.Sequence) or len(ndims) != 2:
            raise ValueError("Invalid 'ndims' definition ({})".format(ndims))

        try:
            cls._check_value(value, ndims)
        except Exception as e:
            return False
        else:
            return True

    @classmethod
    def zeros(cls, ndims=CartesianSpace.ndims):
        return cls([0] * ndims, ndims=(ndims, ndims))

    @classmethod
    def ones(cls, ndims=CartesianSpace.ndims):
        return cls([1] * ndims, ndims=(ndims, ndims))

    @classmethod
    def from_k(cls, value, ndims=CartesianSpace.ndims):
        return cls([value] * ndims, ndims=(ndims, ndims))

    @classmethod
    def from_value(cls, value):
        if isinstance(value, collections.abc.Iterable):
            return cls(list(value))
        else:
            return cls.from_k(value)

    def __new__(cls, sizes, *args, ndims=(1, 3)):
        if len(args) > 0:
            sizes = [sizes, *args]

        if isinstance(ndims, int):
            ndims = tuple([ndims] * 2)
        elif not isinstance(ndims, collections.abc.Sequence) or len(ndims) != 2:
            raise ValueError("Invalid 'ndims' definition ({})".format(ndims))

        try:
            cls._check_value(sizes, ndims=ndims)
        except Exception as e:
            raise TypeError("Invalid {} definition".format(cls.__name__)) from e
        else:
            return super().__new__(cls, sizes)

    def __getattr__(self, name):
        try:
            value = self[CartesianSpace.Axis.symbols.index(name)]
        except (IndexError, ValueError) as e:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            ) from e
        else:
            return value

    def __getitem__(self, item):
        if isinstance(item, CartesianSpace.Axis):
            item = item.value
        return tuple.__getitem__(self, item)

    def __add__(self, other):
        """Element-wise addition."""
        return self._apply(self._broadcast(other), operator.add)

    def __sub__(self, other):
        """Element-wise substraction."""
        return self._apply(self._broadcast(other), operator.sub)

    def __mul__(self, other):
        """Element-wise multiplication."""
        return self._apply(self._broadcast(other), operator.mul)

    def __floordiv__(self, other):
        """"Element-wise integer division."""
        return self._apply(self._broadcast(other), operator.floordiv)

    def __and__(self, other):
        """Element-wise intersection operation.
        """
        return self._apply(other, min)

    def __or__(self, other):
        """Element-wise union operation.
        """
        return self._apply(other, max)

    def __lt__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.lt)

    def __le__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.le)

    def __eq__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.eq)

    def __ne__(self, other):
        """Element-wise comparison.
        """
        return not self._compare(self._broadcast(other), operator.eq)

    def __gt__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.gt)

    def __ge__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.ge)

    def __repr__(self):
        return "{cls_name}({value})".format(
            cls_name=type(self).__name__, value=tuple.__repr__(self)
        )

    def __hash__(self):
        return tuple.__hash__(self)

    def __str__(self):
        return tuple.__repr__(self)

    @property
    def __dict__(self):
        """Ordered mapping from axes names to their values."""
        return collections.OrderedDict(zip(CartesianSpace.Axis.symbols, self))

    @property
    def ndims(self):
        return len(self)

    def union(self, other):
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return self.__or__(other)

    def intersection(self, other):
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return self.__and__(other)

    def _apply(self, other, func):
        if not isinstance(other, type(self)) or len(self) != len(other):
            raise ValueError("Incompatible instance '{obj}'".format(obj=other))

        return type(self)([func(a, b) for a, b in zip(self, other)])

    def _broadcast(self, value):
        if isinstance(value, int):
            value = type(self)([value] * self.ndims)
        elif type(self).is_valid(value, ndims=self.ndims):
            value = type(self)(value)

        return value

    def _compare(self, other, op):
        if len(self) != len(other):  # or not isinstance(other, type(self))
            raise ValueError("Incompatible instance '{obj}'".format(obj=other))

        return all(op(a, b) for a, b in zip(self, other))


class Index(NumericTuple):
    """Index in a grid (all elements are ints)."""

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(isinstance(d, int) for d in value)
        assert ndims[0] <= len(value) <= ndims[1]


class Shape(NumericTuple):
    """Shape of a n-dimensional grid (all elements are int >= 0)."""

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(isinstance(d, int) and d >= 0 for d in value)
        assert ndims[0] <= len(value) <= ndims[1]


class FrameTuple(tuple):
    """N-dimensional list of pairs of numbers representing offsets around one central origin."""

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(
            len(r) == 2 and isinstance(r[0], numbers.Number) and isinstance(r[1], numbers.Number)
            for r in value
        )
        assert ndims[0] <= len(value) <= ndims[1]

    @classmethod
    def is_valid(cls, value, *, ndims=(1, CartesianSpace.ndims)):
        if isinstance(ndims, int):
            ndims = tuple([ndims] * 2)
        elif not isinstance(ndims, collections.abc.Sequence) or len(ndims) != 2:
            raise ValueError("Invalid 'ndims' definition ({})".format(ndims))

        try:
            cls._check_value(value, ndims)
        except Exception as e:
            return False
        else:
            return True

    @classmethod
    def zeros(cls, ndims=CartesianSpace.ndims):
        return cls([(0, 0)] * ndims, ndims=(ndims, ndims))

    @classmethod
    def ones(cls, ndims=CartesianSpace.ndims):
        return cls([(1, 1)] * ndims, ndims=(ndims, ndims))

    @classmethod
    def from_k(cls, value_pair, ndims=CartesianSpace.ndims):
        return cls([value_pair] * ndims, ndims=(ndims, ndims))

    @classmethod
    def from_lower_upper(cls, lower, upper):
        ndims = max(len(lower), len(upper))
        return cls([(lower[i], upper[i]) for i in range(ndims)], ndims=(ndims, ndims))

    def __new__(cls, ranges, *args, ndims=(1, 3)):
        if len(args) > 0:
            ranges = [ranges, *args]
        try:
            cls._check_value(ranges, ndims=ndims)
        except Exception as e:
            raise TypeError("Invalid definition") from e
        else:
            return super().__new__(cls, ranges)

    def __getattr__(self, name):
        try:
            value = self[CartesianSpace.Axis.symbols.index(name)]
        except (IndexError, ValueError) as e:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            ) from e
        else:
            return value

    def __getitem__(self, item):
        if isinstance(item, CartesianSpace.Axis):
            item = item.value
        return tuple.__getitem__(self, item)

    def __add__(self, other):
        """Element-wise addition."""
        # return self._apply(other, lambda a, b: a + b)
        return self._apply(self._broadcast(other), lambda a, b: a + b)

    def __sub__(self, other):
        """Element-wise substraction."""
        return self._apply(self._broadcast(other), lambda a, b: a - b)

    def __and__(self, other):
        """Element-wise intersection operation.
        """
        return self._apply(other, min, min)

    def __or__(self, other):
        """Element-wise union operation.
        """
        return self._apply(other, max, max)

    def __lt__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.lt)

    def __le__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.le)

    def __eq__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.eq)

    def __ne__(self, other):
        """Element-wise comparison.
        """
        return not self._compare(self._broadcast(other), operator.eq)

    def __gt__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.gt)

    def __ge__(self, other):
        """Element-wise comparison.
        """
        return self._compare(self._broadcast(other), operator.ge)

    def __repr__(self):
        return "{cls_name}({value})".format(
            cls_name=self.__class__.__name__, value=tuple.__repr__(self)
        )

    def __hash__(self):
        return tuple.__hash__(self)

    def __str__(self):
        return tuple.__repr__(self)

    @property
    def __dict__(self):
        """Ordered mapping from axes names to their values."""
        return collections.OrderedDict(zip(CartesianSpace.Axis.symbols, self))

    @property
    def ndims(self):
        return len(self)

    @property
    def is_symmetric(self):
        return all(d[0] == d[1] for d in self)

    @property
    def lower_indices(self):
        return tuple(d[0] for d in self)

    @property
    def upper_indices(self):
        return tuple(d[1] for d in self)

    def append(self, point):
        other = self.__class__([(i, i) for i in point])
        return self.__or__(other)

    def concatenate(self, other):
        if not isinstance(other, FrameTuple):
            other = type(self)(other)
        return self.__add__(other)

    def union(self, other):
        if not isinstance(other, FrameTuple):
            other = type(self)(other)
        return self.__or__(other)

    def intersection(self, other):
        if not isinstance(other, FrameTuple):
            other = type(self)(other)
        return self.__and__(other)

    def _apply(self, other, left_func, right_func=None):
        if not isinstance(other, FrameTuple) or len(self) != len(other):
            raise ValueError("Incompatible instance '{obj}'".format(obj=other))

        right_func = right_func or left_func
        return type(self)(
            [tuple([left_func(a[0], b[0]), right_func(a[1], b[1])]) for a, b in zip(self, other)]
        )

    def _reduce(self, reduce_func, out_type=tuple):
        return out_type([reduce_func(d[0], d[1]) for d in self])

    def _compare(self, other, left_op, right_op=None):
        if len(self) != len(other):  # or not isinstance(other, Frame)
            raise ValueError("Incompatible instance '{obj}'".format(obj=other))

        right_op = right_op or left_op
        return all(left_op(a[0], b[0]) and right_op(a[1], b[1]) for a, b in zip(self, other))

    def _broadcast(self, value):
        if isinstance(value, int):
            value = type(self)([(value, value)] * self.ndims)
        elif type(self).is_valid(value, ndims=self.ndims):
            value = type(self)(value)

        return value


class Boundary(FrameTuple):
    """Frame size around one central origin (pairs of integers).

    Negative numbers represent a boundary region substracting from
    the wrapped area.
    """

    __slots__ = ()

    def __hash__(self):
        return tuple.__hash__(self)

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(
            len(r) == 2 and isinstance(r[0], int) and isinstance(r[1], int)
            # and r[0] >= 0 and r[1] >= 0
            for r in value
        )
        assert ndims[0] <= len(value) <= ndims[1]

    @classmethod
    def from_offset(cls, offset):
        if not Index.is_valid(offset):
            raise ValueError("Invalid offset value ({})".format(offset))
        return cls([(-1 * min(0, i), max(0, i)) for i in offset])

    @property
    def frame_size(self):
        """Sizes of the boundary area, i.e., excluding the central element of the frame."""
        return Shape(tuple(d[1] + d[0] for d in self))

    @property
    def shape(self):
        """Total shape of the frame area."""
        return Shape(tuple(d[1] + d[0] + 1 for d in self))


class Extent(FrameTuple):
    """Stencil support: region defined by the smallest and the largest offsets in
     a stencil pattern computation.

    Size of boundary regions are expressed as minimum and maximum relative offsets related
    to the computation position. For example the frame for a stencil accessing 3 elements
    at the left of computation point and two at the right in the X axis would be: (-3, 2).
    If the stencil only has accesses with negative indices, the upper boundary would be then
    a negative value.
    """

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(
            len(r) == 2
            and isinstance(r[0], (int, type(None)))
            and isinstance(r[1], (int, type(None)))
            and (r[0] is None or r[1] is None or r[1] <= r[1])
            for r in value
        )
        assert ndims[0] <= len(value) <= ndims[1]

    @classmethod
    def empty(cls, ndims=CartesianSpace.ndims):
        return cls([(None, None)] * ndims)

    @classmethod
    def from_offset(cls, offset):
        if not Index.is_valid(offset):
            raise ValueError("Invalid offset value ({})".format(offset))
        return cls([(i, i) for i in offset])

    def __and__(self, other):
        """Intersection operation.
        """
        return self._apply(other, max, min)

    def __or__(self, other):
        """Union operation.
        """
        return self._apply(other, min, max)

    def __lt__(self, other):
        return self._compare(other, operator.gt, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.ge, operator.le)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return not self._compare(other, operator.eq)

    def __gt__(self, other):
        return self._compare(other, operator.lt, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.le, operator.ge)

    def __hash__(self):
        return tuple.__hash__(self)

    @property
    def frame_size(self):
        """Sizes of the boundary area, i.e., excluding the central element of the frame."""
        return Shape(tuple(d[1] - d[0] for d in self))

    @property
    def shape(self):
        """Total shape of the frame area."""
        return Shape(tuple(d[1] - d[0] + 1 for d in self))

    def to_boundary(self):
        return Boundary([(-1 * d[0], d[1]) for d in self])

    def _apply(self, other, left_func, right_func=None):
        if not isinstance(other, FrameTuple) or len(self) != len(other):
            raise ValueError("Incompatible instance '{obj}'".format(obj=other))

        right_func = right_func or left_func
        result = [None] * len(self)
        for i, (a, b) in enumerate(zip(self, other)):
            if a[0] is None:
                left = b[0]
            elif b[0] is None:
                left = a[0]
            else:
                left = left_func(a[0], b[0])

            if a[1] is None:
                right = b[1]
            elif b[0] is None:
                right = a[1]
            else:
                right = right_func(a[1], b[1])

            result[i] = (left, right)

        return type(self)(result)


class CenteredExtent(Extent):
    """Stencil support: region defined by the largest negative offset (or zero) and
    the largest positive offset (or zero) in a stencil pattern computation.

    Size of boundary regions are expressed as minimum and maximum relative offsets related
    to the computation position. For example the frame for a stencil accessing 3 elements
    at the left of computation point and two at the right in the X axis would be: (-3, 2).
    If the stencil only has accesses with negative indices, the upper boundary would be 0.
    """

    __slots__ = ()

    @classmethod
    def _check_value(cls, value, ndims):
        assert isinstance(value, collections.abc.Sequence), "Invalid sequence"
        assert all(
            len(r) == 2 and isinstance(r[0], int) and isinstance(r[1], int) and r[0] <= 0 <= r[1]
            for r in value
        )
        assert ndims[0] <= len(value) <= ndims[1]

    @classmethod
    def empty(cls, ndims=CartesianSpace.ndims):
        return cls.zeros(ndims)

    @classmethod
    def from_offset(cls, offset):
        if not Index.is_valid(offset):
            raise ValueError("Invalid offset value ({})".format(offset))
        return cls([(min(i, 0), max(i, 0)) for i in offset])

    def to_boundary(self):
        return Boundary([(-1 * min(d[0], 0), max(d[1], 0)) for d in self])


class Grid:
    pass


class Namespace(dict):
    def __getattr__(self, item):
        return self[item]


@enum.unique
class AccessKind(enum.Enum):
    READ_ONLY = 0
    READ_WRITE = 1

    def __str__(self):
        return self.name


class DomainInfo(
    collections.namedtuple("DomainInfoNamedTuple", ["parallel_axes", "sequential_axis", "ndims"])
):
    pass


class FieldInfo(collections.namedtuple("FieldInfoNamedTuple", ["access", "boundary", "dtype"])):
    def __repr__(self):
        result = "FieldInfo(access=AccessKind.{access}, boundary={boundary}, dtype={dtype})".format(
            access=self.access.name, boundary=repr(self.boundary), dtype=repr(self.dtype)
        )
        return result


class ParameterInfo(collections.namedtuple("ParameterInfoNamedTuple", ["dtype"])):
    pass


@attribclass(frozen=True)
class StencilID:
    qualified_name = attribute(of=str)
    version = attribute(of=str)

    def __iter__(self):
        return iter([self.qualified_name, self.version])


@attribkwclass
class BuildOptions:
    """Build options."""

    name = attribute(of=str)
    module = attribute(of=str)
    backend_opts = attribute(of=DictOf[str, Any], factory=dict)
    build_info = attribute(of=dict, optional=True)
    rebuild = attribute(of=bool, default=False)

    @property
    def qualified_name(self):
        name = ".".join([self.module, self.name])
        return name

    @property
    def shashed_id(self):
        result = gt_utils.shashed_id(
            self.name, self.module, *tuple(sorted(self.backend_opts.items()))
        )

        return result


class StencilObject(abc.ABC):
    """Generic singleton implementation of a stencil function.

    This class is used as base class for the specific subclass generated
    at run-time for any stencil definition and a unique set of external symbols.
    Instances of this class do not contain any information and thus it is
    implemented as a singleton: only one instance per subclass is actually
    allocated (and it is immutable).
    """

    def __new__(cls, *args, **kwargs):
        if getattr(cls, "_instance", None) is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __setattr__(self, key, value):
        raise AttributeError("Attempting a modification of an attribute in a frozen class")

    def __delattr__(self, item):
        raise AttributeError("Attempting a deletion of an attribute in a frozen class")

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        result = """
<StencilObject: {name}> [backend="{backend}"]
    - I/O fields: {fields} 
    - Parameters: {params} 
    - Constants: {constants} 
    - Definition ({func}):
{source} 
        """.format(
            name=self.options["module"] + "." + self.options["name"],
            version=self._gt_id_,
            backend=self.backend,
            fields=self.field_info,
            params=self.parameter_info,
            constants=self.constants,
            func=self.definition_func,
            source=self.source,
        )

        return result

    def __hash__(self):
        return int.from_bytes(type(self)._gt_id_.encode(), byteorder="little")

    # Those attributes are added to the class at loading time:
    #
    #   _gt_id_ (stencil_id.version)
    #   definition_func

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def source(self):
        pass

    @property
    @abc.abstractmethod
    def domain_info(self):
        pass

    @property
    @abc.abstractmethod
    def field_info(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def parameter_info(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def constants(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def options(self) -> dict:
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def call_run(self, field_args, parameter_args, domain, origin, exec_info=None):
        """Check and preprocess the provided arguments (called by :class:`StencilObject` subclasses).

        Note that this function will always try to expand simple parameter values to
        complete data structures by repeating the same value as many times as needed.

        Parameters
        ----------
            field_args: `dict`
                Mapping from field names to actually passed data arrays.
                This parameter encapsulates `*args` in the actual stencil subclass
                by doing: `{input_name[i]: arg for i, arg in enumerate(args)}`

            parameter_args: `dict`
                Mapping from parameter names to actually passed parameter values.
                This parameter encapsulates `**kwargs` in the actual stencil subclass
                by doing: `{name: value for name, value in kwargs.items()}`

            domain : `Sequence` of `int`, optional
                Shape of the computation domain. If `None`, it will be used the
                largest feasible domain according to the provided input fields
                and origin values (`None` by default).

            origin :  `[int * ndims]` or `{'field_name': [int * ndims]}`, optional
                If a single offset is passed, it will be used for all fields.
                If a `dict` is passed, there could be an entry for each field.
                A special key *'_all_'* will represent the value to be used for all
                the fields not explicitly defined. If `None` is passed or it is
                not possible to assign a value to some field according to the
                previous rule, the value will be inferred from the global boundaries
                of the field. Note that the function checks if the origin values
                are at least equal to the `global_border` attribute of that field,
                so a 0-based origin will only be acceptable for fields with
                a 0-area support region.

            exec_info : `dict`, optional
                Dictionary used to store information about the stencil execution.
                (`None` by default).

        Returns
        -------
            `None`

        Raises
        -------
            ValueError
                If invalid data or inconsistent options are specified.
        """

        if exec_info is not None:
            exec_info["call_run_start_time"] = time.perf_counter()
        used_arg_fields = {
            name: field for name, field in field_args.items() if self.field_info[name] is not None
        }
        for name, field_info in self.field_info.items():
            if field_info is not None and field_args[name] is None:
                raise ValueError("Field '{field_name}' is None.".format(field_name=name))
        for name, parameter_info in self.parameter_info.items():
            if parameter_info is not None and parameter_args[name] is None:
                raise ValueError(
                    "Parameter '{parameter_name}' is None.".format(parameter_name=name)
                )
        # assert compatibility of fields with stencil
        for name, field in used_arg_fields.items():
            if not gt.backend.from_name(self.backend).storage_info["is_compatible_layout"](field):
                raise ValueError(
                    "The layout of the field {} is not compatible with the backend.".format(name)
                )
            # ToDo: check if mask is correct: need mask info in stencil object.

            if not field.is_stencil_view:
                raise ValueError(
                    "An incompatible view was passed for field " + name + " to the stencil. "
                )
            for name_other, field_other in used_arg_fields.items():
                if field_other.mask == field.mask:
                    if not field_other.shape == field.shape:
                        raise ValueError(
                            "The fields {} and {} have the same mask but different shapes.".format(
                                name, name_other
                            )
                        )

        assert isinstance(field_args, dict) and isinstance(parameter_args, dict)

        # Shapes
        shapes = {}

        for name, field in used_arg_fields.items():
            # if hasattr(field, "grid_group"):
            #     # Extract ndarray from gt.storage object
            #     field = field.data
            shapes[name] = Shape(field.shape)
        # Origins
        if origin is None:
            origin = {}
        else:
            origin = normalize_origin_mapping(origin)
        for name, field in used_arg_fields.items():
            origin.setdefault(name, origin["_all_"] if "_all_" in origin else field.default_origin)

        # all_origin = Shape(origin["_all_"]) if "_all_" in origin else None
        # for name in field_args.keys():
        #     min_origin = Shape(self.field_info[name].boundary.lower_indices)
        #     if name not in origin:
        #         origin[name] = all_origin if all_origin else min_origin
        #     else:
        #         origin[name] = Shape(origin[name])
        #     if not origin[name] >= min_origin:
        #         raise ValueError(
        #             "Origin value smaller than global boundary for field '{}'".format(name)
        #         )

        # Domain
        max_domain = Shape([sys.maxsize] * self.domain_info.ndims)
        for name, shape in shapes.items():
            upper_boundary = Index(self.field_info[name].boundary.upper_indices)
            max_domain &= shape - (Index(origin[name]) + upper_boundary)

        if domain is None:
            domain = max_domain
        else:
            domain = normalize_domain(domain)
            if len(domain) != self.domain_info.ndims:
                raise ValueError("Invalid 'domain' value ({})".format(domain))

        # check domain+halo vs field size
        if not domain > Shape.zeros(self.domain_info.ndims):
            raise ValueError("Compute domain contains zero sizes ({})".format(domain))

        if not domain <= max_domain:
            raise ValueError(
                "Compute domain too large (provided: {}, maximum: {})".format(domain, max_domain)
            )
        for name, field in used_arg_fields.items():
            min_origin = self.field_info[name].boundary.lower_indices
            if origin[name] < min_origin:
                raise ValueError(
                    "Origin for field {} too small. Must be at least {}, is {}".format(
                        name, min_origin, origin[name]
                    )
                )
            min_shape = tuple(
                o + d + h
                for o, d, h in zip(
                    origin[name], domain, self.field_info[name].boundary.upper_indices
                )
            )
            if min_shape > field.shape:
                raise ValueError(
                    "Shape of field {} is {} but must be at least {} for given domain and origin.".format(
                        name, field.shape, min_shape
                    )
                )

        # if domain != max_domain:
        #     warnings.warn("Input fields do not match default domain size!", UserWarning)

        # field_args = {k: v.view(np.ndarray) for k, v in field_args.items()}

        self.run(
            **field_args, **parameter_args, _domain_=domain, _origin_=origin, exec_info=exec_info
        )


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


def normalize_domain(domain):
    if not isinstance(domain, Shape):
        if not Shape.is_valid(domain):
            raise ValueError("Invalid 'domain' value ({})".format(domain))
        domain = Shape(domain)

    return domain


def normalize_origin(origin):
    if origin is not None:
        if isinstance(origin, int):
            origin = Shape.from_k(origin)
        elif isinstance(origin, collections.abc.Sequence) and Index.is_valid(origin):
            origin = Shape.from_value(origin)
        else:
            raise ValueError("Invalid 'origin' value ({})".format(origin))

    return origin


def normalize_origin_mapping(origin_mapping):
    origin_mapping = origin_mapping or {}
    if isinstance(origin_mapping, collections.abc.Mapping):
        origin_mapping = {
            key: normalize_origin(value)
            for key, value in gt_utils.normalize_mapping(
                origin_mapping, key_types=[str], filter_none=True
            ).items()
        }
    else:
        origin_mapping = {"_all_": normalize_origin(origin_mapping)}

    return origin_mapping
