import abc
import sys
import time
import warnings

import numpy as np

import gt4py.backend as gt_backend
import gt4py.storage as gt_storage
from gt4py.definitions import (
    AccessKind,
    Boundary,
    DomainInfo,
    FieldInfo,
    Index,
    ParameterInfo,
    Shape,
    normalize_domain,
    normalize_origin_mapping,
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

    def _get_max_domain(self, field_args, origin):
        """Return the maximum domain size possible

        Parameters
        ----------
            field_args: `dict`
                Mapping from field names to actually passed data arrays.

            origin: `{'field_name': [int * ndims]}`
                The origin for each field.


        Returns
        -------
            `Shape`: the maximum domain size.
        """
        max_domain = Shape([np.iinfo(np.uintc).max] * self.domain_info.ndims)
        shapes = {name: Shape(field.shape) for name, field in field_args.items()}
        for name, shape in shapes.items():
            upper_boundary = Index(self.field_info[name].boundary.upper_indices)
            max_domain &= shape - (Index(origin[name]) + upper_boundary)
        return max_domain

    def _validate_args(self, used_field_args, used_param_args, domain, origin):
        """Validate input arguments to _call_run.

        Raises
        -------
            ValueError
                If invalid data or inconsistent options are specified.

            TypeError
                If an incorrect field or parameter data type is passed.
        """

        # assert compatibility of fields with stencil
        for name, field in used_field_args.items():
            if not gt_backend.from_name(self.backend).storage_info["is_compatible_layout"](field):
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
            if not field.dtype == self.field_info[name].dtype:
                raise TypeError(
                    f"The dtype of field '{name}' is '{field.dtype}' instead of '{self.field_info[name].dtype}'"
                )
            # ToDo: check if mask is correct: need mask info in stencil object.

            if isinstance(field, gt_storage.storage.Storage):
                if not field.is_stencil_view:
                    raise ValueError(
                        f"An incompatible view was passed for field {name} to the stencil. "
                    )

        # assert compatibility of parameters with stencil
        for name, parameter in used_param_args.items():
            if not type(parameter) == self.parameter_info[name].dtype:
                raise TypeError(
                    f"The type of parameter '{name}' is '{type(parameter)}' instead of '{self.parameter_info[name].dtype}'"
                )

        assert isinstance(used_field_args, dict) and isinstance(used_param_args, dict)

        if len(domain) != self.domain_info.ndims:
            raise ValueError(f"Invalid 'domain' value '{domain}'")

        # check domain+halo vs field size
        if not domain > Shape.zeros(self.domain_info.ndims):
            raise ValueError(f"Compute domain contains zero sizes '{domain}')")

        max_domain = self._get_max_domain(used_field_args, origin)
        if not domain <= max_domain:
            raise ValueError(
                f"Compute domain too large (provided: {domain}, maximum: {max_domain})"
            )
        for name, field in used_field_args.items():
            min_origin = self.field_info[name].boundary.lower_indices
            if origin[name] < min_origin:
                raise ValueError(
                    f"Origin for field {name} too small. Must be at least {min_origin}, is {origin[name]}"
                )
            min_shape = tuple(
                o + d + h
                for o, d, h in zip(
                    origin[name], domain, self.field_info[name].boundary.upper_indices
                )
            )
            if min_shape > field.shape:
                raise ValueError(
                    f"Shape of field {name} is {field.shape} but must be at least {min_shape} for given domain and origin."
                )

    def _call_run(
        self, field_args, parameter_args, domain, origin, *, validate_args=True, exec_info=None
    ):
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

        # Collect used arguments and parameters
        used_field_args = {
            name: field
            for name, field in field_args.items()
            if self.field_info.get(name, None) is not None
        }
        for name, field_info in self.field_info.items():
            if field_info is not None and used_field_args[name] is None:
                raise ValueError(f"Field '{name}' is None.")

        used_param_args = {
            name: param
            for name, param in parameter_args.items()
            if self.parameter_info.get(name, None) is not None
        }
        for name, parameter_info in self.parameter_info.items():
            if parameter_info is not None and used_param_args[name] is None:
                raise ValueError(f"Parameter '{name}' is None.")

        # Origins
        if origin is None:
            origin = {}
        else:
            origin = normalize_origin_mapping(origin)

        for name, field in used_field_args.items():
            origin.setdefault(name, origin["_all_"] if "_all_" in origin else field.default_origin)

        # Domain
        if domain is None:
            domain = self._get_max_domain(used_field_args, origin)
            if any(axis_bound == np.iinfo(np.uintc).max for axis_bound in domain):
                raise ValueError(
                    f"Compute domain could not be deduced. Specifiy the domain explicitly or ensure you reference at least one field."
                )
        else:
            domain = normalize_domain(domain)

        if validate_args:
            self._validate_args(used_field_args, used_param_args, domain, origin)

        self.run(
            _domain_=domain, _origin_=origin, exec_info=exec_info, **field_args, **parameter_args
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()
