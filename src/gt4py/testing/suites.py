# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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
import collections
import inspect
import sys
import types
from itertools import count, product

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest

import gt4py as gt
from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py import utils as gt_utils
from gt4py.stencil_object import StencilObject

from .input_strategies import (
    composite_implementation_strategy_factory,
    composite_strategy_factory,
    ndarray_shape_st,
    ndarray_st,
    padded_shape_st,
)
from .utils import annotate_function, standardize_dtype_dict


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

    required_members = {
        "domain_range",
        "symbols",
        "definition",
        "validation",
        "backends",
        "dtypes",
    }

    def collect_symbols(cls_name, bases, cls_dict):

        domain_range = cls_dict["domain_range"]

        domain_strategy = cls_dict["domain_strategy"] = hyp_st.shared(
            ndarray_shape_st(domain_range), key=cls_name
        )

        generation_strategy_factories = cls_dict["generation_strategy_factories"] = dict()
        implementation_strategy_factories = cls_dict["implementation_strategy_factories"] = dict()
        global_boundaries = cls_dict["global_boundaries"] = dict()
        constants = cls_dict["constants"] = dict()
        singletons = cls_dict["singletons"] = dict()
        max_boundary = ((0, 0), (0, 0), (0, 0))
        for symbol in cls_dict["symbols"].values():
            if symbol.kind == "field":
                max_boundary = tuple(
                    (max(m[0], abs(b[0])), max(m[0], b[0]))
                    for m, b in zip(max_boundary, symbol.boundary)
                )
        cls_dict["max_boundary"] = max_boundary

        for name, symbol in cls_dict["symbols"].items():

            if symbol.kind == "global_strategy":
                generation_strategy_factories[name] = symbol.value_st_factory
            elif symbol.kind == "global_set":
                constants[name] = symbol.values
            elif symbol.kind == "singleton":
                singletons[name] = symbol.values[0]
            elif symbol.kind == "field":
                global_boundaries[name] = symbol.boundary
                shape_strategy = padded_shape_st(
                    domain_strategy, [abs(d[0]) + abs(d[1]) for d in max_boundary]
                )

                # default arguments necessary to avoid late binding
                def implementation_strategy_factory(
                    dt, shape=shape_strategy, value_st_factory=symbol.value_st_factory
                ):
                    return ndarray_st(dt, shape_strategy, value_st_factory)

                implementation_strategy_factories[name] = implementation_strategy_factory
            elif symbol.kind == "parameter":
                implementation_strategy_factories[name] = symbol.value_st_factory
            elif symbol.kind == "none":
                implementation_strategy_factories[name] = symbol.value_st_factory

            else:
                raise AssertionError

        cls_dict["origin"] = tuple(o[0] for o in max_boundary)

    def parametrize_generation_tests(cls_name, bases, cls_dict):

        dtypes = cls_dict["dtypes"]
        backends = cls_dict["backends"]
        generation_strategy_factories = cls_dict["generation_strategy_factories"]

        def get_dtype_combinations(dtypes):
            grouped_combinations = [
                {k: v for k, v in zip(dtypes.keys(), p)} for p in product(*dtypes.values())
            ]
            ret = []
            for combination in grouped_combinations:
                d = dict()
                for ktuple in combination:
                    for k in ktuple:
                        d[k] = combination[ktuple]
                ret.append(d)
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

        parameters = inspect.getfullargspec(cls_dict["definition"]).kwonlyargs
        cls_dict["tests"] = []
        for d in get_dtype_combinations(dtypes):
            for g in get_globals_combinations(d):
                for b in backends:
                    cls_dict["tests"].append(
                        dict(
                            backend=b,
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
                                    k: (
                                        dtype.type
                                        if (k in cls_dict["constants"] or k in parameters)
                                        else gtscript.Field[dtype.type, gtscript.IJK]
                                    )
                                    for k, dtype in d.items()
                                },
                            ),
                        )
                    )

        def generation_test_wrapper(self, test):
            @hyp.given(hypothesis_data=test["generation_strategy"]())
            def hyp_wrapper(test_hyp, hypothesis_data):
                bases[0]._test_generation(
                    self,
                    test_hyp,
                    {**test_hyp["constants"], **hypothesis_data, **cls_dict["singletons"]},
                )

            hyp_wrapper(test)

        pytest_params = []

        for test in cls_dict["tests"]:
            if test["suite"] == cls_name:
                marks = (
                    [pytest.mark.requires_gpu]
                    if gt_backend.from_name(test["backend"]).storage_info["device"] == "gpu"
                    else ()
                )
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

    def parametrize_implementation_tests(cls_name, bases, cls_dict):

        implementation_strategy_factories = cls_dict["implementation_strategy_factories"]
        global_boundaries = cls_dict["global_boundaries"]

        def implementation_test_wrapper(self, test, implementation_strategy):
            @hyp.given(hypothesis_data=implementation_strategy())
            def hyp_wrapper(test_hyp, hypothesis_data):
                bases[0]._test_implementation(self, test_hyp, hypothesis_data)

            hyp_wrapper(test)

        runtime_pytest_params = []
        for test in (t for t in cls_dict["tests"] if t["suite"] == cls_name):
            marks = (
                [pytest.mark.requires_gpu]
                if gt_backend.from_name(test["backend"]).storage_info["device"] == "gpu"
                else ()
            )
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
    def _validate_new_args(cls, cls_name, bases, cls_dict):
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
        assert isinstance(backends, collections.abc.Sequence) and all(
            isinstance(b, str) for b in backends
        ), "'backends' must be a sequence of strings"
        for b in backends:
            assert b in gt.backend.REGISTRY.names, "backend '{backend}' not supported".format(
                backend=b
            )

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

        cls._validate_new_args(cls_name, bases, cls_dict)

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

        cls.collect_symbols(cls_name, bases, cls_dict)

        assert set(input_names + parameter_names) == set(
            cls_dict["implementation_strategy_factories"].keys()
        ), "Missing or invalid keys in 'symbols' mapping (generated: {})".format(
            cls_dict["implementation_strategy_factories"].keys()
        )

        cls.parametrize_generation_tests(cls_name, bases, cls_dict)
        cls.parametrize_implementation_tests(cls_name, bases, cls_dict)

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

                    {
                        'float_symbols' : (np.float32, np.float64),
                        'int_symbols' : (int, np.int_, np.int64)
                    }

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

    def _test_generation(self, test, externals_dict):
        """Test source code generation for all *backends* and *stencil suites*.

        The generated implementations are cached in a :class:`utils.ImplementationsDB`
        instance, to avoid duplication of (potentially expensive) compilations.
        """
        cls = type(self)
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

        # Assert strict equality for Dawn backends
        if implementation.backend.startswith("dawn"):
            assert all(
                field_info.boundary == cls.global_boundaries[name]
                for name, field_info in implementation.field_info.items()
                if field_info is not None
            )
        else:
            assert all(
                field_info.boundary >= cls.global_boundaries[name]
                for name, field_info in implementation.field_info.items()
                if field_info is not None
            )

        test["implementations"].append(implementation)

    def _run_test_implementation(self, parameters_dict, implementation):

        cls = type(self)
        fields, exec_info = parameters_dict

        # Domain
        from gt4py.definitions import Shape
        from gt4py.ir.nodes import Index

        origin = cls.origin
        max_boundary = cls.max_boundary

        shape_iter = (Shape(v.shape) for v in fields.values() if isinstance(v, np.ndarray))
        shape = next(shape_iter)
        assert all(shape == sh for sh in shape_iter)

        max_domain = Shape([sys.maxsize] * implementation.domain_info.ndims)
        for name in (k for k, v in fields.items() if isinstance(v, np.ndarray)):
            upper_boundary = Index(implementation.field_info[name].boundary.upper_indices)
            max_domain &= shape - (Index(origin) + upper_boundary)
        domain = max_domain
        for k, v in implementation.constants.items():
            sys.modules[self.__module__].__dict__[k] = v

        inputs = {}
        for k, f in fields.items():
            if isinstance(f, np.ndarray):
                inputs[k] = gt_storage.from_array(
                    f,
                    dtype=f.dtype,
                    shape=shape,
                    default_origin=origin,
                    backend=implementation.backend,
                )

            else:
                inputs[k] = f

        # remove unused input parameters
        inputs = {key: value for key, value in inputs.items() if value is not None}

        validation_fields = {}
        validation_inputs = {}
        for name, field in inputs.items():
            if name in implementation.field_info:
                field_extent_low = implementation.field_info[name].boundary.lower_indices
                field_extent_high = implementation.field_info[name].boundary.upper_indices
                field_origin = tuple(o - e for o, e in zip(origin, field_extent_low))
                field_extent_high = tuple(b[1] - e for b, e in zip(max_boundary, field_extent_high))
                validation_slice = tuple(
                    slice(o, s - h) for o, s, h in zip(field_origin, shape, field_extent_high)
                )
                validation_fields[name] = np.array(field, copy=True)
                validation_inputs[name] = validation_fields[name][validation_slice]
            else:
                validation_inputs[name] = field

        implementation(**inputs, origin=origin, exec_info=exec_info)
        domain = exec_info["domain"]

        cls.validation(
            **validation_inputs,
            domain=domain,
            origin={
                name: implementation.field_info[name].boundary.lower_indices
                for name in validation_fields
                if name in implementation.field_info
            },
        )

        # Test values
        for (name, value), expected_value in zip(inputs.items(), validation_fields.values()):
            if isinstance(fields[name], np.ndarray):

                if gt_backend.from_name(value.backend).storage_info["device"] == "gpu":
                    value.synchronize()
                    value = value.data.get()
                else:
                    value = value.data

                np.testing.assert_allclose(
                    value,
                    expected_value,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=EQUAL_NAN,
                    err_msg="Wrong data in output field '{name}'".format(name=name),
                )

    def _test_implementation(self, test, parameters_dict):
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

            self._run_test_implementation(parameters_dict, implementation)
