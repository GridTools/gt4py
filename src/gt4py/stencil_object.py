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
import collections.abc
import sys
import time
import typing
import warnings
from dataclasses import dataclass
from pickle import dumps
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union

import numpy as np

import gt4py.backend as gt_backend
import gt4py.storage as gt_storage
import gtc.utils as gtc_utils
from gt4py.definitions import AccessKind, DomainInfo, FieldInfo, ParameterInfo
from gtc.definitions import Index, Shape


FieldType = Union[gt_storage.storage.Storage, np.ndarray]
OriginType = Union[Tuple[int, int, int], Dict[str, Tuple[int, ...]]]


def _compute_cache_key(field_args, parameter_args, domain, origin) -> int:
    # field.default_origin is computed using getattr to support numpy.ndarray.
    field_data = tuple(
        (name, field.shape, getattr(field, "default_origin", (0, 0, 0)))
        for name, field in field_args.items()
        if field is not None
    )
    return hash((field_data, *parameter_args.keys(), dumps(domain), dumps(origin)))


@dataclass(frozen=True)
class FrozenStencil:
    """Stencil with pre-computed domain and origin for each field argument."""

    stencil_object: "StencilObject"
    origin: Dict[str, Tuple[int, ...]]
    domain: Tuple[int, ...]

    def __post_init__(self):
        for name, field_info in self.stencil_object.field_info.items():
            if name not in self.origin or len(self.origin[name]) != field_info.ndim:
                raise ValueError(
                    f"'{name}' origin {self.origin.get(name)} is not a {field_info.ndim}-dimensional integer tuple"
                )

    def __call__(self, **kwargs) -> None:
        assert "origin" not in kwargs and "domain" not in kwargs
        exec_info = kwargs.get("exec_info")

        if exec_info is not None:
            exec_info["call_run_start_time"] = time.perf_counter()

        field_args = {name: kwargs[name] for name in self.stencil_object.field_info.keys()}
        parameter_args = {name: kwargs[name] for name in self.stencil_object.parameter_info.keys()}

        self.stencil_object.run(
            _domain_=self.domain,
            _origin_=self.origin,
            exec_info=exec_info,
            **field_args,
            **parameter_args,
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()

    def __sdfg__(self, **kwargs):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.stencil_object.backend}")'
        )

    def __sdfg_signature__(self):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.stencil_object.backend}")'
        )

    def __sdfg_closure__(self, *args, **kwargs):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.stencil_object.backend}")'
        )


class StencilObject(abc.ABC):
    """Generic singleton implementation of a stencil callable.

    This class is used as base class for specific subclass generated
    at run-time for any stencil definition and a unique set of external symbols.
    Instances of this class do not contain state and thus it is
    implemented as a singleton: only one instance per subclass is actually
    allocated (and it is immutable).

    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments.

    Keyword Arguments
    ------------------
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
        previous rule, the value will be inferred from the `boundary` attribute
        of the `field_info` dict. Note that the function checks if the origin values
        are at least equal to the `boundary` attribute of that field,
        so a 0-based origin will only be acceptable for fields with
        a 0-area support region.

    exec_info : `dict`, optional
        Dictionary used to store information about the stencil execution.
        (`None` by default). If the dictionary contains the magic key
        '__aggregate_data' and it evaluates to `True`, the dictionary is
        populated with a nested dictionary per class containing different
        performance statistics. These include the stencil calls count, the
        cumulative time spent in all stencil calls, and the actual time spent
        in carrying out the computations.

    """

    # Those attributes are added to the class at loading time:
    _gt_id_: str
    definition_func: Callable[..., Any]

    _domain_origin_cache: ClassVar[Dict[int, Tuple[Tuple[int, ...], Dict[str, Tuple[int, ...]]]]]
    """Stores domain/origin pairs that have been used by hash."""

    def __new__(cls, *args, **kwargs):
        if getattr(cls, "_instance", None) is None:
            cls._instance = object.__new__(cls)
            cls._domain_origin_cache = {}
        return cls._instance

    def __setattr__(self, key, value) -> None:
        raise AttributeError("Attempting a modification of an attribute in a frozen class")

    def __delattr__(self, item) -> None:
        raise AttributeError("Attempting a deletion of an attribute in a frozen class")

    def __eq__(self, other) -> bool:
        return type(self) == type(other)

    def __str__(self) -> str:
        result = """
<StencilObject: {name}> [backend="{backend}"]
    - I/O fields: {fields}
    - Parameters: {params}
    - Constants: {constants}
    - Version: {version}
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

    def __hash__(self) -> int:
        return int.from_bytes(type(self)._gt_id_.encode(), byteorder="little")

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def source(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def domain_info(self) -> DomainInfo:
        pass

    @property
    @abc.abstractmethod
    def field_info(self) -> Dict[str, FieldInfo]:
        pass

    @property
    @abc.abstractmethod
    def parameter_info(self) -> Dict[str, ParameterInfo]:
        pass

    @property
    @abc.abstractmethod
    def constants(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def options(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def _make_origin_dict(origin: Any) -> Dict[str, Index]:
        try:
            if isinstance(origin, dict):
                return dict(origin)
            if origin is None:
                return {}
            if isinstance(origin, collections.abc.Iterable):
                return {"_all_": Index.from_value(origin)}
            if isinstance(origin, int):
                return {"_all_": Index.from_k(origin)}
        except Exception:
            pass

        raise ValueError("Invalid 'origin' value ({})".format(origin))

    @staticmethod
    def _get_max_domain(
        field_args: Dict[str, Any],
        domain_infos: DomainInfo,
        field_infos: Dict[str, FieldInfo],
        origin: Dict[str, Tuple[int, ...]],
        *,
        squeeze: bool = True,
    ) -> Shape:
        """Return the maximum domain size possible.

        Parameters
        ----------
            field_args:
                Mapping from field names to actually passed data arrays.
            origin:
                The origin for each field.
            squeeze:
                Convert non-used domain dimensions to singleton dimensions.

        Returns
        -------
            `Shape`: the maximum domain size.
        """
        domain_ndim = domain_infos.ndim
        max_size = sys.maxsize
        max_domain = Shape([max_size] * domain_ndim)

        for name, field_info in field_infos.items():
            if field_info.access != AccessKind.NONE:
                assert field_args.get(name, None) is not None, f"Invalid value for '{name}' field."
                field = field_args[name]
                api_domain_mask = field_info.domain_mask
                api_domain_ndim = field_info.domain_ndim
                upper_indices = field_info.boundary.upper_indices.filter_mask(api_domain_mask)
                field_origin = Index.from_value(origin[name])
                field_domain = tuple(
                    field.shape[i] - (field_origin[i] + upper_indices[i])
                    for i in range(api_domain_ndim)
                )
                max_domain &= Shape.from_mask(field_domain, api_domain_mask, default=max_size)

        if squeeze:
            return Shape([i if i != max_size else 1 for i in max_domain])
        else:
            return max_domain

    def _validate_args(  # noqa: C901  # Function is too complex
        self,
        field_args: Dict[str, FieldType],
        param_args: Dict[str, Any],
        domain: Tuple[int, ...],
        origin: Dict[str, Tuple[int, ...]],
    ) -> None:
        """
        Validate input arguments to _call_run.

        Raises
        -------
            ValueError
                If invalid data or inconsistent options are specified.

            TypeError
                If an incorrect field or parameter data type is passed.
        """
        assert isinstance(field_args, dict) and isinstance(param_args, dict)

        # validate domain sizes
        domain_ndim = self.domain_info.ndim
        if len(domain) != domain_ndim:
            raise ValueError(f"Invalid 'domain' value '{domain}'")

        try:
            domain = Shape(domain)
        except Exception:
            raise ValueError("Invalid 'domain' value ({})".format(domain))

        if not domain > Shape.zeros(domain_ndim):
            raise ValueError(f"Compute domain contains zero sizes '{domain}')")

        if not domain <= (
            max_domain := self._get_max_domain(
                field_args, self.domain_info, self.field_info, origin, squeeze=False
            )
        ):
            raise ValueError(
                f"Compute domain too large (provided: {domain}, maximum: {max_domain})"
            )

        if domain[2] < self.domain_info.min_sequential_axis_size:
            raise ValueError(
                f"Compute domain too small. Sequential axis is {domain[2]}, but must be at least {self.domain_info.min_sequential_axis_size}."
            )

        # assert compatibility of fields with stencil
        for name, field_info in self.field_info.items():
            if field_info.access != AccessKind.NONE:
                if name not in field_args:
                    raise ValueError(f"Missing value for '{name}' field.")
                field = field_args[name]

                if not gt_backend.from_name(self.backend).storage_info["is_compatible_layout"](
                    field
                ):
                    raise ValueError(
                        f"The layout of the field {name} is not compatible with the backend."
                    )

                if not gt_backend.from_name(self.backend).storage_info["is_compatible_type"](field):
                    raise ValueError(
                        f"Field '{name}' has type '{type(field)}', which is not compatible with the '{self.backend}' backend."
                    )
                elif type(field) is np.ndarray:
                    warnings.warn(
                        "NumPy ndarray passed as field. This is discouraged and only works with constraints and only for certain backends.",
                        RuntimeWarning,
                    )

                field_dtype = self.field_info[name].dtype
                if not field.dtype == field_dtype:
                    raise TypeError(
                        f"The dtype of field '{name}' is '{field.dtype}' instead of '{field_dtype}'"
                    )

                if isinstance(field, gt_storage.storage.Storage) and not field.is_stencil_view:
                    raise ValueError(
                        f"An incompatible view was passed for field {name} to the stencil. "
                    )

                # Check: domain + halo vs field size
                field_info = self.field_info[name]
                field_domain_mask = field_info.domain_mask
                field_domain_ndim = field_info.domain_ndim
                field_domain_origin = Index.from_mask(origin[name], field_domain_mask[:domain_ndim])

                if field.ndim != field_domain_ndim + len(field_info.data_dims):
                    raise ValueError(
                        f"Storage for '{name}' has {field.ndim} dimensions but the API signature "
                        f"expects {field_domain_ndim + len(field_info.data_dims)} ('{field_info.axes}[{field_info.data_dims}]')"
                    )

                if (
                    isinstance(field, gt_storage.storage.Storage)
                    and tuple(field.mask)[:domain_ndim] != field_domain_mask
                ):
                    raise ValueError(
                        f"Storage for '{name}' has domain mask '{field.mask}' but the API signature "
                        f"expects '[{', '.join(field_info.axes)}]'"
                    )

                # Check: data dimensions shape
                if field.shape[field_domain_ndim:] != field_info.data_dims:
                    raise ValueError(
                        f"Field '{name}' expects data dimensions {field_info.data_dims} but got {field.shape[field_domain_ndim:]}"
                    )

                min_origin = gtc_utils.interpolate_mask(
                    field_info.boundary.lower_indices.filter_mask(field_domain_mask),
                    field_domain_mask,
                    default=0,
                )
                if field_domain_origin < min_origin:
                    raise ValueError(
                        f"Origin for field {name} too small. Must be at least {min_origin}, is {field_domain_origin}"
                    )

                spatial_domain = typing.cast(Shape, domain).filter_mask(field_domain_mask)
                lower_indices = field_info.boundary.lower_indices.filter_mask(field_domain_mask)
                upper_indices = field_info.boundary.upper_indices.filter_mask(field_domain_mask)
                min_shape = tuple(
                    lb + d + ub for lb, d, ub in zip(lower_indices, spatial_domain, upper_indices)
                )
                if min_shape > field.shape:
                    raise ValueError(
                        f"Shape of field {name} is {field.shape} but must be at least {min_shape} for given domain and origin."
                    )

        # assert compatibility of parameters with stencil
        for name, parameter_info in self.parameter_info.items():
            if parameter_info.access != AccessKind.NONE:
                if name not in param_args:
                    raise ValueError(f"Missing value for '{name}' parameter.")
                elif type(parameter := param_args[name]) != parameter_info.dtype:
                    raise TypeError(
                        f"The type of parameter '{name}' is '{type(parameter)}' instead of '{parameter_info.dtype}'"
                    )

    @staticmethod
    def _normalize_origins(
        field_args: Dict[str, FieldType],
        field_infos: Dict[str, FieldInfo],
        origin: Optional[OriginType],
    ) -> Dict[str, Tuple[int, ...]]:
        origin = StencilObject._make_origin_dict(origin)
        all_origin = origin.get("_all_", None)
        # Set an appropriate origin for all fields

        for name, field_info in field_infos.items():
            assert name in field_args, f"Missing value for '{name}' field."
            field_origin = origin.get(name, None)

            if field_origin is not None:
                field_origin_ndim = len(field_origin)
                if field_origin_ndim != field_info.ndim:
                    assert (
                        field_origin_ndim == field_info.domain_ndim
                    ), f"Invalid origin specification ({field_origin}) for '{name}' field."
                    origin[name] = (*field_origin, *((0,) * len(field_info.data_dims)))

            elif all_origin is not None:
                origin[name] = (
                    *gtc_utils.filter_mask(all_origin, field_info.domain_mask),
                    *((0,) * len(field_info.data_dims)),
                )

            elif hasattr(field_arg := field_args.get(name), "default_origin"):
                assert field_arg is not None  # for mypy
                origin[name] = field_arg.default_origin
            else:
                origin[name] = (0,) * field_info.ndim

        return origin

    def _call_run(
        self,
        field_args: Dict[str, FieldType],
        parameter_args: Dict[str, Any],
        domain: Optional[Tuple[int, ...]],
        origin: Optional[OriginType],
        *,
        validate_args: bool = True,
        exec_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Check and preprocess the provided arguments (called by :class:`StencilObject` subclasses).

        Note that this function will always try to expand simple parameter values to complete
        data structures by repeating the same value as many times as needed.

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

        Check :class:`StencilObject` for a full specification of the `domain`,
        `origin` and `exec_info` keyword arguments.

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

        cache_key = _compute_cache_key(field_args, parameter_args, domain, origin)
        if cache_key not in self._domain_origin_cache:
            origin = self._normalize_origins(field_args, self.field_info, origin)

            if domain is None:
                domain = self._get_max_domain(field_args, self.domain_info, self.field_info, origin)

            if validate_args:
                self._validate_args(field_args, parameter_args, domain, origin)

            type(self)._domain_origin_cache[cache_key] = (domain, origin)
        else:
            domain, origin = type(self)._domain_origin_cache[cache_key]

        self.run(
            _domain_=domain, _origin_=origin, exec_info=exec_info, **field_args, **parameter_args
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()

    def freeze(
        self: "StencilObject", *, origin: Dict[str, Tuple[int, ...]], domain: Tuple[int, ...]
    ) -> FrozenStencil:
        """Return a StencilObject wrapper with a fixed domain and origin for each argument.

        Parameters
        ----------
            origin: `dict`
                The origin for each Field argument.

            domain: `Sequence` of `int`
                The compute domain shape for the frozen stencil.

        Notes
        ------
        Both `origin` and `domain` arguments should be compatible with the domain and origin
        specification defined in :class:`StencilObject`.

        Returns
        -------
            `FrozenStencil`
                The stencil wrapper. This should be called with the regular stencil arguments,
                but the field origins and domain cannot be changed. Note, no checking of origin
                or domain occurs at call time so it is the users responsibility to ensure
                correct usage.
        """
        return FrozenStencil(self, origin, domain)

    def clean_call_args_cache(self: "StencilObject") -> None:
        """Clean the argument cache.

        Returns
        -------
            None
        """
        type(self)._domain_origin_cache.clear()

    def __deepcopy__(self, memodict=None):
        # StencilObjects are singletons.
        return self

    def __sdfg__(self, *args, **kwargs):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.backend}")'
        )

    def __sdfg_signature__(self):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.backend}")'
        )

    def __sdfg_closure__(self, *args, **kwargs):
        raise TypeError(
            f'Only dace backends are supported in DaCe-orchestrated programs. (found "{self.backend}")'
        )
