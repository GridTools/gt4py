# -*- coding: utf-8 -*-
import abc
import collections.abc
import sys
import time
import warnings
from typing import Any, Collection, Dict, Tuple

import numpy as np

import gt4py.backend as gt_backend
import gt4py.ir as gt_ir
import gt4py.storage as gt_storage
import gt4py.utils as gt_utils
from gt4py.definitions import (
    AccessKind,
    Boundary,
    CartesianSpace,
    DomainInfo,
    FieldInfo,
    Index,
    ParameterInfo,
    Shape,
)


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
        except:
            pass

        raise ValueError("Invalid 'origin' value ({})".format(origin))

    def _get_max_domain(
        self,
        field_args: Dict[str, Any],
        origin: Dict[str, Tuple[int, ...]],
        *,
        squeeze: bool = True,
    ) -> Shape:
        """Return the maximum domain size possible

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
        domain_ndim = self.domain_info.ndim
        max_size = sys.maxsize
        max_domain = Shape([max_size] * domain_ndim)

        for name, field_info in self.field_info.items():
            if field_info is not None:
                assert field_args.get(name, None) is not None, f"Invalid value for '{name}' field."
                field = field_args[name]
                api_domain_mask = field_info.domain_mask
                api_domain_ndim = field_info.domain_ndim
                assert (
                    not isinstance(field, gt_storage.storage.Storage)
                    or tuple(field.mask)[:domain_ndim] == api_domain_mask
                ), (
                    f"Storage for '{name}' has domain mask '{field.mask}' but the API signature "
                    f"expects '[{', '.join(field_info.axes)}]'"
                )
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

    def _validate_args(self, field_args, param_args, domain, origin) -> None:
        """Validate input arguments to _call_run.

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
        except:
            raise ValueError("Invalid 'domain' value ({})".format(domain))

        if not domain > Shape.zeros(domain_ndim):
            raise ValueError(f"Compute domain contains zero sizes '{domain}')")

        if not domain <= (max_domain := self._get_max_domain(field_args, origin, squeeze=False)):
            raise ValueError(
                f"Compute domain too large (provided: {domain}, maximum: {max_domain})"
            )

        # assert compatibility of fields with stencil
        for name, field_info in self.field_info.items():
            if field_info is not None:
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

                min_origin = gt_utils.interpolate_mask(
                    field_info.boundary.lower_indices.filter_mask(field_domain_mask),
                    field_domain_mask,
                    default=0,
                )
                if field_domain_origin < min_origin:
                    raise ValueError(
                        f"Origin for field {name} too small. Must be at least {min_origin}, is {field_domain_origin}"
                    )

                spatial_domain = domain.filter_mask(field_domain_mask)
                upper_indices = field_info.boundary.upper_indices.filter_mask(field_domain_mask)
                min_shape = tuple(
                    o + d + h for o, d, h in zip(field_domain_origin, spatial_domain, upper_indices)
                )
                if min_shape > field.shape:
                    raise ValueError(
                        f"Shape of field {name} is {field.shape} but must be at least {min_shape} for given domain and origin."
                    )

        # assert compatibility of parameters with stencil
        for name, parameter_info in self.parameter_info.items():
            if parameter_info is not None:
                if name not in param_args:
                    raise ValueError(f"Missing value for '{name}' parameter.")
                if not type(parameter := param_args[name]) == self.parameter_info[name].dtype:
                    raise TypeError(
                        f"The type of parameter '{name}' is '{type(parameter)}' instead of '{self.parameter_info[name].dtype}'"
                    )

    def _call_run(
        self, field_args, parameter_args, domain, origin, *, validate_args=True, exec_info=None
    ) -> None:
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

            origin :  `[int * ndims]` or {'field_name': [int * ndims]} , optional
                If a single offset is passed, it will be used for all fields.
                If a `dict` is passed, there could be an entry for each field.
                A special key '_all_' will represent the value to be used for all
                the fields not explicitly defined. If `None` is passed or it is
                not possible to assign a value to some field according to the
                previous rule, the value will be inferred from the global boundaries
                of the field. Note that the function checks if the origin values
                are at least equal to the `global_border` attribute of that field,
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

        domain_ndim = self.domain_info.ndim
        origin = self._make_origin_dict(origin)
        all_origin = origin.get("_all_", None)

        # Set an appropriate origin for all fields
        for name, field_info in self.field_info.items():
            if field_info is not None:
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
                        *gt_utils.filter_mask(all_origin, field_info.domain_mask),
                        *((0,) * len(field_info.data_dims)),
                    )

                elif isinstance(field_arg := field_args[name], gt_storage.storage.Storage):
                    origin[name] = field_arg.default_origin

                else:
                    origin[name] = (0,) * field_info.ndim

        # Domain
        if domain is None:
            domain = self._get_max_domain(field_args, origin)

        assert (
            len(domain) == domain_ndim
        ), f"Provided domain '{domain}' is not {domain_ndim}-dimensional."

        if validate_args:
            self._validate_args(field_args, parameter_args, domain, origin)

        self.run(
            _domain_=domain, _origin_=origin, exec_info=exec_info, **field_args, **parameter_args
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()
