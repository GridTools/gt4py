# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import inspect
import sys
import types
from itertools import count, product

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest

import gt4py.cartesian.gtc.utils as gtc_utils
from gt4py import cartesian as gt, cartesian as gt4pyc, storage as gt_storage
from gt4py.cartesian import gtscript, utils as gt_utils
from gt4py.cartesian.definitions import AccessKind, FieldInfo
from gt4py.cartesian.gtc.definitions import Boundary, CartesianSpace, Index, Shape
from gt4py.cartesian.stencil_object import StencilObject
from gt4py.storage.cartesian import utils as storage_utils

from .input_strategies import (
    SymbolKind,
    composite_implementation_strategy_factory,
    composite_strategy_factory,
    derived_shape_st,
    ndarray_shape_st,
    ndarray_st,
)
from .utils import annotate_function, standardize_dtype_dict


ParameterSet = type(pytest.param())

counter = count()
RTOL = 1e-05
ATOL = 1e-08
EQUAL_NAN = False

unique_str_ctr = count()


def unique_str():
    return str(next(unique_str_ctr))


class SuiteMeta(type):
    """Custom metaclass for all :class:`StencilTestSuite` classes.

    This metaclass generates a namespace-like class where all methods are static
    and adds new members with the required testing strategies and expected results
    to test a stencil definition and implementations using Hypothesis and pytest.
    """

    required_members = {"domain_range", "symbols", "definition", "validation", "backends", "dtypes"}

    def collect_symbols(cls_name, cls_dict):
        domain_range = cls_dict["domain_range"]
        domain_strategy = cls_dict["domain_strategy"] = hyp_st.shared(
            ndarray_shape_st(domain_range), key=cls_name
        )

        generation_strategy_factories = cls_dict["generation_strategy_factories"] = dict()
        implementation_strategy_factories = cls_dict["implementation_strategy_factories"] = dict()
        global_boundaries = cls_dict["global_boundaries"] = dict()
        constants = cls_dict["constants"] = dict()
        singletons = cls_dict["singletons"] = dict()
        cls_dict["field_params"] = field_params = {}
        max_boundary = ((0, 0), (0, 0), (0, 0))
        for symbol in cls_dict["symbols"].values():
            if symbol.kind == SymbolKind.FIELD:
                max_boundary = tuple(
                    (max(m[0], b[0]), max(m[1], b[1]))
                    for m, b in zip(max_boundary, symbol.boundary)
                )
        cls_dict["max_boundary"] = max_boundary

        for name, symbol in cls_dict["symbols"].items():
            if symbol.kind == SymbolKind.GLOBAL_STRATEGY:
                generation_strategy_factories[name] = symbol.value_st_factory
            elif symbol.kind == SymbolKind.GLOBAL_SET:
                constants[name] = symbol.values
            elif symbol.kind == SymbolKind.SINGLETON:
                singletons[name] = symbol.values[0]
            elif symbol.kind == SymbolKind.FIELD:
                if symbol.axes:
                    axes = symbol.axes
                    extra_shape = tuple(
                        b[0] + b[1] if ax in symbol.axes else None
                        for b, ax in zip(max_boundary, "IJK")
                    )
                else:
                    axes = "IJK"
                    extra_shape = tuple(b[0] + b[1] for b in max_boundary)

                if symbol.data_dims:
                    data_dims = symbol.data_dims
                    extra_shape = (*extra_shape, *symbol.data_dims)
                else:
                    data_dims = tuple()

                field_params[name] = (axes, data_dims)

                global_boundaries[name] = symbol.boundary
                shape_strategy = derived_shape_st(domain_strategy, extra_shape)

                # Use default arguments to pass values avoiding late binding problems
                def implementation_strategy_factory(
                    dt, shape=shape_strategy, value_st_factory=symbol.value_st_factory
                ):
                    return ndarray_st(dt, shape, value_st_factory)

                implementation_strategy_factories[name] = implementation_strategy_factory
            elif symbol.kind == SymbolKind.PARAMETER:
                implementation_strategy_factories[name] = symbol.value_st_factory
            elif symbol.kind == SymbolKind.NONE:
                implementation_strategy_factories[name] = symbol.value_st_factory

            else:
                raise AssertionError

        cls_dict["origin"] = tuple(o[0] for o in max_boundary)

    def parametrize_generation_tests(cls_name, cls_dict):
        backends = cls_dict["backends"]
        dtypes = cls_dict["dtypes"]
        field_params = cls_dict["field_params"]
        generation_strategy_factories = cls_dict["generation_strategy_factories"]

        def get_dtype_combinations(dtypes):
            grouped_combinations = [
                {k: v for k, v in zip(dtypes.keys(), p)} for p in product(*dtypes.values())
            ]
            ret = [
                {k: combination[ktuple] for ktuple in combination for k in ktuple}
                for combination in grouped_combinations
            ]
            return ret

        def get_globals_combinations(dtypes):
            combinations = [
                {k: dtypes[k].type(v) for k, v in zip(cls_dict["constants"].keys(), p)}
                for p in product(*cls_dict["constants"].values())
            ]
            if not combinations:
                return [{}]
            else:
                return combinations

        cls_dict["tests"] = []
        for b in backends:
            for d in get_dtype_combinations(dtypes):
                for g in get_globals_combinations(d):
                    cls_dict["tests"].append(
                        dict(
                            backend=b if isinstance(b, str) else b.values[0],
                            marks=[] if isinstance(b, str) else b.marks,
                            suite=cls_name,
                            constants=g,
                            dtypes=d,
                            generation_strategy=composite_strategy_factory(
                                d, generation_strategy_factories
                            ),
                            implementations=[],
                            test_id=len(cls_dict["tests"]),
                            definition=annotate_function(
                                function=cls_dict["definition"],
                                dtypes={
                                    name: (
                                        dtype.type
                                        if (name not in field_params)
                                        else gtscript.Field[
                                            getattr(gtscript, field_params[name][0]),
                                            (dtype.type, field_params[name][1]),
                                        ]
                                    )
                                    for name, dtype in d.items()
                                },
                            ),
                        )
                    )

        def generation_test_wrapper(self, test):
            @hyp.given(hypothesis_data=test["generation_strategy"]())
            def hyp_wrapper(test_hyp, hypothesis_data):
                self._test_generation(
                    test_hyp, {**test_hyp["constants"], **hypothesis_data, **cls_dict["singletons"]}
                )

            hyp_wrapper(test)

        pytest_params = []

        for test in cls_dict["tests"]:
            if test["suite"] == cls_name:
                marks = test["marks"]
                if gt4pyc.backend.from_name(test["backend"]).storage_info["device"] == "gpu":
                    marks.append(pytest.mark.requires_gpu)
                name = test["backend"]
                name += "".join(f"_{key}_{value}" for key, value in test["constants"].items())
                name += "".join(
                    "_{}_{}".format(key, value.name) for key, value in test["dtypes"].items()
                )
                param = pytest.param(test, marks=marks, id=name)
                pytest_params.append(param)

        cls_dict["test_generation"] = pytest.mark.parametrize("test", pytest_params)(
            generation_test_wrapper
        )

    def parametrize_implementation_tests(cls_name, cls_dict):
        implementation_strategy_factories = cls_dict["implementation_strategy_factories"]
        global_boundaries = cls_dict["global_boundaries"]

        def implementation_test_wrapper(self, test, implementation_strategy):
            @hyp.given(hypothesis_data=implementation_strategy())
            def hyp_wrapper(test_hyp, hypothesis_data):
                self._test_implementation(test_hyp, hypothesis_data)

            hyp_wrapper(test)

        runtime_pytest_params = []
        for test in cls_dict["tests"]:
            if test["suite"] == cls_name:
                marks = test["marks"]
                if gt4pyc.backend.from_name(test["backend"]).storage_info["device"] == "gpu":
                    marks.append(pytest.mark.requires_gpu)
                name = test["backend"]
                name += "".join(f"_{key}_{value}" for key, value in test["constants"].items())
                name += "".join(
                    "_{}_{}".format(key, value.name) for key, value in test["dtypes"].items()
                )
                runtime_pytest_params.append(
                    pytest.param(
                        test,
                        composite_implementation_strategy_factory(
                            test["dtypes"], implementation_strategy_factories, global_boundaries
                        ),
                        marks=marks,
                        id=name,
                    )
                )

        cls_dict["test_implementation"] = pytest.mark.parametrize(
            ("test", "implementation_strategy"), runtime_pytest_params
        )(implementation_test_wrapper)

    @classmethod
    def _validate_new_args(cls, cls_name, cls_dict):
        missing_members = cls.required_members - cls_dict.keys()
        if len(missing_members) > 0:
            raise TypeError(
                "Missing {missing} required members in '{name}' definition".format(
                    missing=missing_members, name=cls_name
                )
            )
        # Check class dict
        domain_range = cls_dict["domain_range"]
        backends = cls_dict["backends"]

        # Create testing strategies
        assert isinstance(cls_dict["symbols"], collections.abc.Mapping), "Invalid 'symbols' mapping"

        # Check domain and ndims
        assert 1 <= len(domain_range) <= 3 and all(
            len(d) == 2 for d in domain_range
        ), "Invalid 'domain_range' definition"

        if any(cls_name.endswith(suffix) for suffix in ("1D", "2D", "3D")):
            assert cls_dict["ndims"] == int(
                cls_name[-2:-1]
            ), "Suite name does not match the actual 'ndims'"

        # Check dtypes
        assert isinstance(
            cls_dict["dtypes"], (collections.abc.Sequence, collections.abc.Mapping)
        ), "'dtypes' must be a sequence or a mapping object"

        # Check backends
        if not all(
            isinstance(b, str)
            or (isinstance(b, ParameterSet) and len(b.values) == 1 and isinstance(b.values[0], str))
            for b in backends
        ):
            raise TypeError("'backends' must be a sequence of strings")
        backends = [pytest.param(b) if isinstance(b, str) else b for b in backends]
        for b in backends:
            if b.values[0] not in gt.backend.REGISTRY.names:
                raise ValueError("backend '{backend}' not supported".format(backend=b))

        # Check definition and validation functions
        if not isinstance(cls_dict["definition"], types.FunctionType):
            raise TypeError("The 'definition' attribute must be a stencil definition function")
        if not isinstance(cls_dict["validation"], types.FunctionType):
            raise TypeError("The 'validation' attribute must be a validation function")

    def __new__(cls, cls_name, bases, cls_dict):
        if cls_dict.get("_skip_", False):  # skip metaclass magic
            return super().__new__(cls, cls_name, bases, cls_dict)
        # Grab members inherited from base classes

        missing_members = cls.required_members - cls_dict.keys()

        for key in missing_members:
            for base in bases:
                if hasattr(base, key):
                    cls_dict[key] = getattr(base, key)
                    break

        dtypes = cls_dict["dtypes"]
        if isinstance(dtypes, collections.abc.Sequence):
            dtypes = {tuple(cls_dict["symbols"].keys()): dtypes}
        cls_dict["dtypes"] = standardize_dtype_dict(dtypes)
        cls_dict["ndims"] = len(cls_dict["domain_range"])

        # Filter out unsupported backends
        cls_dict["backends"] = [
            backend
            for backend in cls_dict["backends"]
            if gt4pyc.backend.from_name(backend if isinstance(backend, str) else backend.values[0])
            is not None
        ]

        cls._validate_new_args(cls_name, cls_dict)

        # Extract input and parameter names
        input_names = []
        parameter_names = []
        definition_signature = inspect.signature(cls_dict["definition"])
        validation_signature = inspect.signature(cls_dict["validation"])
        for (def_name, def_pobj), (val_name, val_pobj) in zip(
            definition_signature.parameters.items(), validation_signature.parameters.items()
        ):
            if def_name != val_name or def_pobj.kind != val_pobj.kind:
                raise ValueError(
                    "Incompatible signatures for 'definition' and 'validation' functions"
                )

            if def_pobj.kind == inspect.Parameter.KEYWORD_ONLY:
                parameter_names.append(def_name)
                if def_pobj.default != inspect.Parameter.empty:
                    assert def_pobj.default == val_pobj.default
            else:
                input_names.append(def_name)

        cls.collect_symbols(cls_name, cls_dict)

        assert set(input_names + parameter_names) == set(
            cls_dict["implementation_strategy_factories"].keys()
        ), "Missing or invalid keys in 'symbols' mapping (generated: {})".format(
            cls_dict["implementation_strategy_factories"].keys()
        )

        cls.parametrize_generation_tests(cls_name, cls_dict)
        cls.parametrize_implementation_tests(cls_name, cls_dict)

        return super().__new__(cls, cls_name, bases, cls_dict)


class StencilTestSuite(metaclass=SuiteMeta):
    """Base class for every *stencil test suite*.

    Every new test suite must inherit from this class and define proper
    attributes and methods to generate a valid set of testing strategies.
    For compatibility with pytest, suites must have names starting with 'Test'

    Supported and required class attributes are:

    Attributes
    ----------
    dtypes : `dict` or `list`
        Required class attribute.
        GlobalDecl suite dtypes dictionary.
        - ``label``: `list` of `dtype`.
        If this value is a `list`, it will be converted to a `dict` with the default
        `None` key assigned to its value. It is meant to be populated with labels representing
        groups of symbols that should have the same type.

        Example:

        .. code-block:: python

                    {"float_symbols": (np.float32, np.float64), "int_symbols": (int, np.int_, np.int64)}

    domain_range : `Sequence` of pairs like `((int, int), (int, int) ... )`
        Required class attribute.
        CartesianSpace sizes for testing. Each item encodes the (min, max) range of sizes for every axis.
    symbols : `dict`
        Required class attribute.
        Definition of symbols (globals, parameters and inputs) used in this stencil.
        - ``name``: `utils._SymbolDescriptor`. It is recommended to use the convenience
        functions `global_name()`, `parameter()` and `field()`. These functions have similar
        and self-explanatory arguments. Note that `dtypes` could be a list of actual dtypes
        but it is usually assumed to be a label from the global suite dtypes dictionary.
    definition : `function`
        Required class attribute.
        Stencil definition function.
    validation : `function`
        Required class attribute.
        Stencil validation function. It should have exactly the same signature than
        arguments to access the actual values used in the current testing invocation.
        the ``definition`` function plus the extra ``_globals_``, ``_domain_``, and ``_origin_``
        It should always return a `list` of `numpy.ndarray` s, one per output, even if
        the function only defines one output value.
    definition_strategies : `dict`
        Automatically generated.
        Hypothesis strategies for the stencil parameters used at definition (externals)
        - ``constant_name``: Hypothesis strategy (`strategy`).
    validation_strategies : `dict`
        Automatically generated.
        Hypothesis strategies for the stencil parameters used at run-time (fields and parameters)
        - ``field_name``: Hypothesis strategy (`strategy`).
        - ``parameter_name``: Hypothesis strategy (`strategy`).
    ndims : `int`
        Automatically generated.
        Constant of dimensions (1-3). If the name of class ends in ["1D", "2D", "3D"],
        this attribute needs to match the name or an assertion error will be raised.
    global_boundaries : `dict`
        Automatically generated.
        Expected global boundaries for the input fields.
        - ``field_name``: 'list' of ``ndim`` 'tuple`s  (``(lower_boundary, upper_boundary)``).
        Example (3D): `[(1, 3), (2, 2), (0, 0)]`
    """

    _skip_ = True  # Avoid processing of this empty test suite

    @classmethod
    def _test_generation(cls, test, externals_dict):
        """Test source code generation for all *backends* and *stencil suites*.

        The generated implementations are cached in a :class:`utils.ImplementationsDB`
        instance, to avoid duplication of (potentially expensive) compilations.
        """
        backend_slug = gt_utils.slugify(test["backend"], valid_symbols="")
        implementation = gtscript.stencil(
            backend=test["backend"],
            definition=test["definition"],
            name=cls.__module__ + f".{test['suite']}_{backend_slug}_{test['test_id']}",
            rebuild=True,
            externals=externals_dict,
        )

        for k, v in externals_dict.items():
            implementation.constants[k] = v

        assert isinstance(implementation, StencilObject)
        assert implementation.backend == test["backend"]

        for name, field_info in implementation.field_info.items():
            if field_info.access == AccessKind.NONE:
                continue
            for i, ax in enumerate("IJK"):
                assert (
                    ax not in field_info.axes
                    or ax == "K"
                    or field_info.boundary[i] >= cls.global_boundaries[name][i]
                )
        test["implementations"].append(implementation)

    @classmethod
    def _run_test_implementation(cls, parameters_dict, implementation):  # too complex
        input_data, exec_info = parameters_dict

        origin = cls.origin
        max_boundary = Boundary(cls.max_boundary)
        field_params = cls.field_params
        field_dimensions = {}
        field_masks = {}
        for name, value in input_data.items():
            if isinstance(value, np.ndarray):
                field_masks[name] = tuple(
                    ax in field_params[name][0] for ax in CartesianSpace.names
                )
                field_dimensions[name] = tuple(
                    ax for ax in CartesianSpace.names if ax in field_params[name][0]
                )

        data_shape = Shape((sys.maxsize,) * 3)
        for name, data in input_data.items():
            if isinstance(data, np.ndarray):
                data_shape &= Shape(
                    gtc_utils.interpolate_mask(data.shape, field_masks[name], default=sys.maxsize)
                )

        domain = data_shape - (
            Index(max_boundary.lower_indices) + Index(max_boundary.upper_indices)
        )

        referenced_inputs = {
            name: info for name, info in implementation.field_info.items() if info is not None
        }
        referenced_inputs.update(
            {name: info for name, info in implementation.parameter_info.items() if info is not None}
        )

        # set externals for validation method
        for k, v in implementation.constants.items():
            sys.modules[cls.__module__].__dict__[k] = v

        # copy input data
        test_values = {}
        validation_values = {}
        for name, data in input_data.items():
            data = input_data[name]
            if name in referenced_inputs:
                info = referenced_inputs[name]
                if isinstance(info, FieldInfo):
                    data_dims = field_params[name][1]
                    if data_dims:
                        dtype = (data.dtype, data_dims)
                    else:
                        dtype = data.dtype
                    test_values[name] = gt_storage.from_array(
                        data=data,
                        dtype=dtype,
                        dimensions=field_dimensions[name],
                        aligned_index=gtc_utils.filter_mask(origin, field_masks[name]),
                        backend=implementation.backend,
                    )
                    validation_values[name] = np.array(data)
                else:
                    test_values[name] = data
                    validation_values[name] = data
            else:
                test_values[name] = None
                validation_values[name] = None

        # call implementation
        implementation(**test_values, origin=origin, domain=domain, exec_info=exec_info)

        # for validation data, data is cropped to actually touched domain, so that origin offsetting
        # does not have to be implemented for every test suite. This is done based on info
        # specified in test suite
        cropped_validation_values = {}
        for name, data in validation_values.items():
            sym = cls.symbols[name]
            if data is not None and sym.kind == SymbolKind.FIELD:
                field_extent_low = tuple(b[0] for b in sym.boundary)
                offset_low = tuple(b[0] - e for b, e in zip(max_boundary, field_extent_low))
                field_extent_high = tuple(b[1] for b in sym.boundary)
                offset_high = tuple(b[1] - e for b, e in zip(max_boundary, field_extent_high))
                validation_slice = gtc_utils.filter_mask(
                    tuple(slice(o, s - h) for o, s, h in zip(offset_low, data_shape, offset_high)),
                    field_masks[name],
                )
                data_dims = field_params[name][1]
                if data_dims:
                    validation_slice = tuple([*validation_slice] + [slice(None)] * len(data_dims))
                cropped_validation_values[name] = data[validation_slice]
            else:
                cropped_validation_values[name] = data

        cls.validation(
            **cropped_validation_values,
            domain=domain,
            origin={
                name: info.boundary.lower_indices
                for name, info in implementation.field_info.items()
                if info is not None
            },
        )

        # Test values
        for name, value in test_values.items():
            if isinstance(value, np.ndarray):
                expected_value = validation_values[name]
                value = storage_utils.cpu_copy(value)

                np.testing.assert_allclose(
                    value,
                    expected_value,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=EQUAL_NAN,
                    err_msg="Wrong data in output field '{name}'".format(name=name),
                )

    @classmethod
    def _test_implementation(cls, test, parameters_dict):
        """Test computed values for implementations generated for all *backends* and *stencil suites*.

        The generated implementations are reused from previous tests by means of a
        :class:`utils.ImplementationsDB` instance shared at module scope.
        """
        implementation_list = test["implementations"]
        if not implementation_list:
            pytest.skip(
                "Cannot perform validation tests, since there are no valid implementations."
            )
        for implementation in implementation_list:
            if not isinstance(implementation, StencilObject):
                raise RuntimeError("Wrong function got from implementations_db cache!")

            cls._run_test_implementation(parameters_dict, implementation)
