import numpy as np

import gt4py
from gt4py import gtscript
from gt4py import storage as gt_store
from tests.test_integration.stencil_definitions import REGISTRY as stencil_registry
from tests.test_integration.test_cpp_regression import get_reference, generate_test_module


def run_horizontal_diffusion(niter, domain, backend, dtype, backend_opts={}):
    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}

    import gt4py.gtscript

    Field3D = gt4py.gtscript.Field[dtype]

    def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
        with computation(PARALLEL), interval(...):
            lap_field = 4.0 * in_field[0, 0, 0] - (
                in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
            )
            res1 = lap_field[1, 0, 0] - lap_field[0, 0, 0]
            flx_field = 0 if (res1 * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res1
            res2 = lap_field[0, 1, 0] - lap_field[0, 0, 0]
            fly_field = 0 if (res2 * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res2
            out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
                flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
            )

    reference_module = gtscript.stencil(
        backend="numpy", definition=horizontal_diffusion, enforce_dtype=dtype
    )
    test_module = gtscript.stencil(
        backend=backend, definition=horizontal_diffusion, enforce_dtype=dtype, **backend_opts,
    )

    validate_shapes = {
        k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()
    }
    arg_fields_reference = get_reference(
        "horizontal_diffusion",
        gt4py.backend.from_name("numpy"),
        domain=domain,
        origins=origins,
        shapes=validate_shapes,
        dtype=dtype,
    )
    arg_fields_test = get_reference(
        "horizontal_diffusion",
        gt4py.backend.from_name(backend),
        domain=domain,
        origins=origins,
        shapes=validate_shapes,
        dtype=dtype,
    )
    arg_fields_reference = {
        k: v for k, v in arg_fields_reference.items() if not k.endswith("_reference")
    }
    arg_fields_test = {k: v for k, v in arg_fields_test.items() if not k.endswith("_reference")}

    for k in arg_fields_reference.keys():
        arg_fields_reference[k].host_to_device()
        arg_fields_test[k].host_to_device()
    reference_module.run(**arg_fields_reference, _domain_=domain, _origin_=origins, exec_info=None)
    test_module.run(**arg_fields_test, _domain_=domain, _origin_=origins, exec_info=None)
    for k in arg_fields_reference.keys():
        arg_fields_reference[k].device_to_host(force=True)
        arg_fields_test[k].device_to_host(force=True)
        if not np.allclose(
            arg_fields_test[k].view(np.ndarray), arg_fields_reference[k].view(np.ndarray)
        ):
            raise RuntimeError(
                "Large error in field {k}. (||err||=={err}, max_atol={atol}, max_rtol={rtol})".format(
                    k=k,
                    err=np.linalg.norm(
                        arg_fields_test[k].view(np.ndarray)
                        - arg_fields_reference[k].view(np.ndarray)
                    ),
                    atol=np.max(
                        np.abs(
                            arg_fields_test[k].view(np.ndarray)
                            - arg_fields_reference[k].view(np.ndarray)
                        )
                    ),
                    rtol=np.max(
                        np.abs(
                            arg_fields_test[k].view(np.ndarray)
                            - arg_fields_reference[k].view(np.ndarray)
                        )
                        / np.abs(arg_fields_reference[k].view(np.ndarray))
                    ),
                )
            )
    del arg_fields_reference
    del arg_fields_test

    perftest_shapes = {
        k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()
    }
    arg_fields_test = {
        name: np.random.randn(*shape).astype(dtype) for name, shape in perftest_shapes.items()
    }
    for k, field in arg_fields_test.items():
        arg_fields_test[k] = gt_store.from_array(
            field,
            dtype=dtype,
            default_origin=origins[k],
            shape=perftest_shapes[k],
            backend=backend,
        )
    for k in arg_fields_test.keys():
        arg_fields_test[k].host_to_device()

    exec_infos = []
    for i in range(niter):
        exec_info = {}
        test_module.run(**arg_fields_test, _domain_=domain, _origin_=origins, exec_info=exec_info)
        exec_infos.append(exec_info)

    return exec_infos


def run_vertical_advection(
    niter, domain, backend, dtype, backend_opts={},
):
    origins = {
        "utens_stage": (0, 0, 0),
        "u_stage": (0, 0, 0),
        "wcon": (0, 0, 0),
        "u_pos": (0, 0, 0),
        "utens": (0, 0, 0),
    }

    import gt4py.gtscript

    Field3D = gt4py.gtscript.Field[dtype]

    def vertical_advection_dycore(
        utens_stage: Field3D,
        u_stage: Field3D,
        wcon: Field3D,
        u_pos: Field3D,
        utens: Field3D,
        *,
        dtr_stage: dtype,
    ):
        from __externals__ import BET_M, BET_P

        with computation(FORWARD):
            with interval(0, 1):
                gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
                cs = gcv * BET_M

                ccol = gcv * BET_P
                bcol = dtr_stage - ccol[0, 0, 0]

                # update the d column
                correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
                dcol = (
                    dtr_stage * u_pos[0, 0, 0]
                    + utens[0, 0, 0]
                    + utens_stage[0, 0, 0]
                    + correction_term
                )

                # Thomas forward
                divided = 1.0 / bcol[0, 0, 0]
                ccol = ccol[0, 0, 0] * divided
                dcol = dcol[0, 0, 0] * divided

            with interval(1, -1):
                gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

                as_ = gav * BET_M
                cs = gcv * BET_M

                acol = gav * BET_P
                ccol = gcv * BET_P
                bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

                # update the d column
                correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (
                    u_stage[0, 0, 1] - u_stage[0, 0, 0]
                )
                dcol = (
                    dtr_stage * u_pos[0, 0, 0]
                    + utens[0, 0, 0]
                    + utens_stage[0, 0, 0]
                    + correction_term
                )

                # Thomas forward
                divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
                ccol = ccol[0, 0, 0] * divided
                dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

            with interval(-1, None):
                gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                as_ = gav * BET_M
                acol = gav * BET_P
                bcol = dtr_stage - acol[0, 0, 0]

                # update the d column
                correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
                dcol = (
                    dtr_stage * u_pos[0, 0, 0]
                    + utens[0, 0, 0]
                    + utens_stage[0, 0, 0]
                    + correction_term
                )

                # Thomas forward
                divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
                dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with computation(BACKWARD):
            with interval(-1, None):
                datacol = dcol[0, 0, 0]
                data_col = datacol
                utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

            with interval(0, -1):
                datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
                data_col = datacol
                utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

    reference_module = gtscript.stencil(
        backend="numpy",
        definition=vertical_advection_dycore,
        externals={"BET_M": 0.5, "BET_P": 0.5},
        enforce_dtype=dtype,
    )
    test_module = gtscript.stencil(
        backend=backend,
        definition=vertical_advection_dycore,
        externals={"BET_M": 0.5, "BET_P": 0.5},
        enforce_dtype=dtype,
        **backend_opts,
    )

    validate_shapes = {
        k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()
    }
    validate_shapes["wcon"] = tuple(
        s + 1 if i == 0 else s for i, s in enumerate(validate_shapes["wcon"])
    )
    arg_fields_reference = get_reference(
        "vertical_advection_dycore",
        gt4py.backend.from_name("numpy"),
        domain=domain,
        origins=origins,
        shapes=validate_shapes,
        dtype=dtype,
    )
    arg_fields_test = get_reference(
        "vertical_advection_dycore",
        gt4py.backend.from_name(backend),
        domain=domain,
        origins=origins,
        shapes=validate_shapes,
        dtype=dtype,
    )
    arg_fields_reference = {
        k: v for k, v in arg_fields_reference.items() if not k.endswith("_reference")
    }
    arg_fields_test = {k: v for k, v in arg_fields_test.items() if not k.endswith("_reference")}

    for k in arg_fields_reference.keys():
        if hasattr(arg_fields_reference[k], "host_to_device"):
            arg_fields_reference[k].host_to_device()
            arg_fields_test[k].host_to_device()

    reference_module.run(**arg_fields_reference, _domain_=domain, _origin_=origins, exec_info=None)
    test_module.run(**arg_fields_test, _domain_=domain, _origin_=origins, exec_info=None)
    for k in arg_fields_reference.keys():
        if not np.isscalar(arg_fields_test[k]):
            arg_fields_reference[k].device_to_host(force=True)
            arg_fields_test[k].device_to_host(force=True)
            if not np.allclose(
                arg_fields_test[k].view(np.ndarray), arg_fields_reference[k].view(np.ndarray)
            ):
                raise RuntimeError(
                    "Large error in field {k}. (||err||=={err}, max_atol={atol}, max_rtol={rtol})".format(
                        k=k,
                        err=np.linalg.norm(
                            arg_fields_test[k].view(np.ndarray)
                            - arg_fields_reference[k].view(np.ndarray)
                        ),
                        atol=np.max(
                            np.abs(
                                arg_fields_test[k].view(np.ndarray)
                                - arg_fields_reference[k].view(np.ndarray)
                            )
                        ),
                        rtol=np.max(
                            np.abs(
                                arg_fields_test[k].view(np.ndarray)
                                - arg_fields_reference[k].view(np.ndarray)
                            )
                            / np.abs(arg_fields_reference[k].view(np.ndarray))
                        ),
                    )
                )
    del arg_fields_reference
    del arg_fields_test

    perftest_shapes = {
        k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()
    }
    perftest_shapes["wcon"] = tuple(
        d + 1 if i == 0 else d for i, d in enumerate(perftest_shapes["wcon"])
    )
    arg_fields_test = {
        name: np.random.randn(*shape).astype(dtype) for name, shape in perftest_shapes.items()
    }
    for k, field in arg_fields_test.items():
        arg_fields_test[k] = gt_store.from_array(
            field,
            dtype=dtype,
            default_origin=origins[k],
            shape=perftest_shapes[k],
            backend=backend,
        )
    arg_fields_test["dtr_stage"] = dtype(np.random.randn(1))

    for k in arg_fields_test.keys():
        if hasattr(arg_fields_test[k], "host_to_device"):
            arg_fields_test[k].host_to_device()

    exec_infos = []
    for i in range(niter):
        exec_info = {}
        test_module.run(**arg_fields_test, _domain_=domain, _origin_=origins, exec_info=exec_info)
        exec_infos.append(exec_info)

    return exec_infos


def summary(exec_infos):
    assert exec_infos
    res = dict(run_time=0.0)
    if "start_run_cpp_time" in exec_infos[0]:
        res["cpp_time"] = 0.0
        res["cpp_time_inner"] = 0.0
    if "pyext_program_start_time" in exec_infos[0]:
        res["pyext_time"] = 0
    for info in exec_infos:
        if "pyext_time" in res:
            res["pyext_time"] += info["pyext_program_end_time"] - info["pyext_program_start_time"]
        res["run_time"] += info["run_end_time"] - info["run_start_time"]
        if "cpp_time" in res:
            res["cpp_time"] += info["end_run_cpp_time"] - info["start_run_cpp_time"]
            res["cpp_time_inner"] += (
                info["end_run_cpp_time_inner"] - info["start_run_cpp_time_inner"]
            )
    return res


from gt4py.backend.dace.base_backend import SDFGInjector


class SpecializingInjector(SDFGInjector):
    def __init__(self, sdfg, domain):
        super().__init__(sdfg)
        self.domain = domain

    def transform_optimize(self, sdfg):
        import copy
        import dace

        res: dace.SDFG = copy.deepcopy(self.sdfg)

        strides = [None] * 3
        layout = gt4py.backend.from_name("adhoc").storage_info["layout_map"]((True, True, True))
        print(layout)
        stride = 1
        domain = [dace.symbolic.symbol(d) for d in "IJK"]
        for i in reversed(np.argsort(layout)):
            strides[i] = stride
            stride = stride * domain[i]

        for var, d in zip("IJK", strides):
            res.replace(f"_ccol_{var}_stride", str(d))
            res.replace(f"_dcol_{var}_stride", str(d))

        if len(res.signature_arglist()) != len(sdfg.signature_arglist()):
            raise ValueError(
                "SDFG to inject does not have matching signature length. ({here} != {raw})".format(
                    here=len(res.signature_arglist()), raw=len(sdfg.signature_arglist())
                )
            )
        if not all(
            here == res for here, res in zip(res.signature_arglist(), sdfg.signature_arglist())
        ):
            l = list(
                (here, res)
                for here, res in zip(res.signature_arglist(), sdfg.signature_arglist())
                if here != res
            )
            raise ValueError(
                "SDFG to inject does not have matching signature. ({here} != {raw})".format(
                    here=l[0][0], raw=l[0][1]
                )
            )
        res.specialize({var: d for var, d in zip("IJK", self.domain)})
        dace.Config.set("compiler", "cuda", "default_block_size", value="128, 1, 1")
        return res


if __name__ == "__main__":
    niter = 1000
    domain = (128, 128, 80)
    data_layout = (1, 2, 0)
    function = "vertical_advection"
    dtype = np.float32
    backend = "adhoc"
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = None
    optimizer = (
        SpecializingInjector(path, domain)
        if path is not None
        else gt4py.backend.dace.cpu_backend.X86DaceOptimizer()
    )
    from gt4py.backend import register as register_backend
    from gt4py.backend.dace.base_backend import DaceBackend
    from gt4py.backend.dace.cpu_backend import CPUDaceBackend
    from gt4py.backend.dace.gpu_backend import GPUDaceBackend

    @register_backend
    # class AdHocBackend(CPUDaceBackend):
    class AdHocBackend(GPUDaceBackend):
        name = "adhoc"
        storage_info = {
            "alignment": 1,  # will not affect temporaries currently
            "device": "gpu",  # change me
            "layout_map": lambda m: data_layout,
            "is_compatible_layout": lambda m: True,
            "is_compatible_type": lambda m: True,
        }
        DEFAULT_OPTIMIZER = optimizer

    print(f"start {backend}")
    backend_opts = dict(rebuild=True)
    exec_infos = globals()[f"run_{function}"](
        niter=niter, domain=domain, backend=backend, dtype=dtype, backend_opts=backend_opts,
    )

    print("times:")
    for k, v in summary(exec_infos).items():
        print("\t{}: {}us per call".format(k, int(v * 1e6 / niter)))
