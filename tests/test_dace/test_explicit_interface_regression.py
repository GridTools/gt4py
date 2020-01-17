import itertools
import numpy as np

import pytest
import hypothesis as hyp
from hypothesis import strategies as hyp_st

from gt4py import storage as gt_store
from gt4py import backend as gt_backend

from ..reference_cpp_regression import reference_module
from .dace_stencil_definitions import REGISTRY as STENCIL_REGISTRY
from .dace_stencil_definitions import build_dace_stencil
from gt4py import definitions as gt_definitions

REGISTRY = list()


def register(cpp_regression_test):
    REGISTRY.append(cpp_regression_test)
    return cpp_regression_test


def get_reference(test_name, backend, domain, origins, shapes, masks=None):

    reference_data = reference_module.__dict__[test_name](*domain)

    res = {}
    for k, data in reference_data.items():
        if np.isscalar(data):
            res[k] = np.float_(data)
        else:
            try:
                field = gt_store.from_array(
                    data,
                    dtype=np.float_,
                    default_origin=origins[k],
                    shape=shapes[k],
                    backend=backend,
                )
            except KeyError:
                field = gt_store.from_array(
                    data,
                    dtype=np.float_,
                    default_origin=origins[k[: -len("_reference")]],
                    shape=shapes[k[: -len("_reference")]],
                    backend=backend,
                )

            res[k] = field
    return res


@register
@hyp.given(domain=hyp_st.tuples(*([hyp_st.integers(min_value=1, max_value=8)] * 3)))
def run_horizontal_diffusion(backend, domain):

    validate_field_names = ["out_field"]
    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "horizontal_diffusion"

    arg_fields = get_reference(name, backend, domain, origins, shapes)

    # testmodule = generate_test_module("horizontal_diffusion")
    testmodule = build_dace_stencil(
        name, gt_definitions.BuildOptions(name=name, module="_test_module." + name)
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()

    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule.run(
        **arg_fields,
        # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        if hasattr(arg_fields[k], "synchronize"):
            arg_fields[k].device_to_host(force=True)
        np.testing.assert_allclose(
            arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
        )


@register
@hyp.given(
    domain=hyp_st.tuples(
        *(
            [hyp_st.integers(min_value=1, max_value=32)] * 2
            + [hyp_st.integers(min_value=2, max_value=32)]
        )
    )
)
def run_tridiagonal_solver(backend, domain):

    validate_field_names = ["out"]
    origins = {
        "inf": (0, 0, 0),
        "diag": (0, 0, 0),
        "sup": (0, 0, 0),
        "rhs": (0, 0, 0),
        "out": (0, 0, 0),
    }
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "tridiagonal_solver"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = build_dace_stencil(
        name, gt_definitions.BuildOptions(name=name, module="_test_module." + name)
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()

    testmodule.run(
        **arg_fields,
        # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        if hasattr(arg_fields[k], "synchronize"):
            arg_fields[k].device_to_host(force=True)
        np.testing.assert_allclose(
            arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
        )


#
# @register
# @hyp.given(
#     domain=hyp_st.tuples(
#         *(
#             [hyp_st.integers(min_value=1, max_value=32)] * 2
#             + [hyp_st.integers(min_value=2, max_value=32)]
#         )
#     )
# )
# def run_vertical_advection_dycore(backend, domain):
#
#     validate_field_names = ["utens_stage"]
#     origins = {
#         "utens_stage": (0, 0, 0),
#         "u_stage": (0, 0, 0),
#         "wcon": (0, 0, 0),
#         "u_pos": (0, 0, 0),
#         "utens": (0, 0, 0),
#     }
#     shapes = {
#         "utens_stage": domain,
#         "u_stage": domain,
#         "wcon": tuple(d + 1 if i == 0 else d for i, d in enumerate(domain)),
#         "u_pos": domain,
#         "utens": domain,
#     }
#     name = "vertical_advection_dycore"
#
#     arg_fields = get_reference(name, backend, domain, origins, shapes)
#     validate_fields = {
#         name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
#     }
#
#     testmodule = generate_test_module("vertical_advection_dycore",)
#     for k in arg_fields:
#         if hasattr(arg_fields[k], "host_to_device"):
#             arg_fields[k].host_to_device()
#     testmodule(**arg_fields, I=np.int32(domain[0]), J=np.int32(domain[1]), K=np.int32(domain[2]))
#
#     for k in validate_field_names:
#         if hasattr(arg_fields[k], "synchronize"):
#             arg_fields[k].device_to_host(force=True)
#         np.testing.assert_allclose(
#             arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
#         )
#


@pytest.mark.parametrize(
    ["backend", "function"], itertools.product(["dace"], REGISTRY,),
)
def test_cpp_regression_cpu(backend, function):
    function(backend)


#
# @pytest.mark.requires_gpu
# @pytest.mark.parametrize(
#     ["backend", "function"],
#     itertools.product(
#         [
#             gt_backend.from_name(name)
#             for name in gt_backend.REGISTRY.names
#             if gt_backend.from_name(name).storage_info["device"] == "gpu"
#         ],
#         REGISTRY,
#     ),
# )
# def test_cpp_regression_gpu(backend, id_version, function):
#     function(backend, id_version)
