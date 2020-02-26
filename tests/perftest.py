from tests.test_integration.stencil_definitions import REGISTRY as stencil_registry
from tests.test_integration.test_cpp_regression import get_reference, generate_test_module


def run_horizontal_diffusion(niter, domain, backend):
    validate_field_names = ["out_field"]
    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "horizontal_diffusion"
    testmodule = generate_test_module(
        "horizontal_diffusion", backend, rebuild=True, id_version="1"
    )
    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()

    exec_infos = []
    for i in range(niter):
        exec_info = {}
        # arg_fields = get_reference(name, backend, domain, origins, shapes)
        # validate_fields = {
        #     name + "_reference": arg_fields.pop(name + "_reference")
        #     for name in validate_field_names
        # }

        testmodule.run(**arg_fields, _domain_=domain, _origin_=origins, exec_info=exec_info)
        exec_infos.append(exec_info)

    return exec_infos


def run_vertical_advection(niter, domain, backend):
    validate_field_names = ["utens_stage"]
    origins = {
        "utens_stage": (0, 0, 0),
        "u_stage": (0, 0, 0),
        "wcon": (0, 0, 0),
        "u_pos": (0, 0, 0),
        "utens": (0, 0, 0),
    }
    shapes = {
        "utens_stage": domain,
        "u_stage": domain,
        "wcon": tuple(d + 1 if i == 0 else d for i, d in enumerate(domain)),
        "u_pos": domain,
        "utens": domain,
    }
    name = "vertical_advection_dycore"
    testmodule = generate_test_module(
        "vertical_advection_dycore", backend, rebuild=True, id_version="1"
    )
    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()

    exec_infos = []
    for i in range(niter):
        exec_info = {}
        # arg_fields = get_reference(name, backend, domain, origins, shapes)
        # validate_fields = {
        #     name + "_reference": arg_fields.pop(name + "_reference")
        #     for name in validate_field_names
        # }

        testmodule.run(**arg_fields, _domain_=domain, _origin_=origins, exec_info=exec_info)
        exec_infos.append(exec_info)

    return exec_infos


def summary(exec_infos):
    assert exec_infos
    res = dict(pyext_time=0.0, run_time=0.0)
    if "start_run_cpp_time" in exec_infos[0]:
        res["cpp_time"] = 0.0
        res["cpp_time_inner"] = 0.0
    for info in exec_infos:
        res["pyext_time"] += info["pyext_program_end_time"] - info["pyext_program_start_time"]
        res["run_time"] += info["run_end_time"] - info["run_start_time"]
        if "cpp_time" in res:
            res["cpp_time"] += info["end_run_cpp_time"] - info["start_run_cpp_time"]
            res["cpp_time_inner"] += (
                info["end_run_cpp_time_inner"] - info["start_run_cpp_time_inner"]
            )

    return res


if __name__ == "__main__":
    import gt4py.backend as gt_backend

    niter = 10
    domain = (256, 256, 64)
    # domain = (16, 16, 32)

    print("##vertical advection")
    print("start dace")
    dace_exec_infos = run_vertical_advection(
        niter=niter, domain=domain, backend=gt_backend.from_name("dacex86")
    )
    print("start gt")
    gt_exec_infos = run_vertical_advection(
        niter=niter, domain=domain, backend=gt_backend.from_name("gtmc")
    )

    print("dace times:")
    for k, v in summary(dace_exec_infos).items():
        print("\t{}: {}".format(k, v))
    print("gt times:")
    for k, v in summary(gt_exec_infos).items():
        print("\t{}: {}".format(k, v))

    # # print("start dacecuda")
    # # dace_exec_infos = run_vertical_advection(
    # #     niter=niter, domain=domain, backend=gt_backend.from_name("dacecuda")
    # # )
    # # print("start gtcuda")
    # # gt_exec_infos = run_vertical_advection(
    # #     niter=niter, domain=domain, backend=gt_backend.from_name("gtcuda")
    # # )
    # #
    # # print("dace times:")
    # # for k, v in summary(dace_exec_infos).items():
    # #     print("\t{}: {}".format(k, v))
    # # print("gt times:")
    # # for k, v in summary(gt_exec_infos).items():
    # #     print("\t{}: {}".format(k, v))

    ##########################################################

    print("##horizontal diffusion")
    print("start dace")
    dace_exec_infos = run_horizontal_diffusion(
        niter=niter, domain=domain, backend=gt_backend.from_name("dacex86")
    )
    print("start gt")
    gt_exec_infos = run_horizontal_diffusion(
        niter=niter, domain=domain, backend=gt_backend.from_name("gtmc")
    )
    print("dace times:")
    for k, v in summary(dace_exec_infos).items():
        print("\t{}: {}".format(k, v))
    print("gt times:")
    for k, v in summary(gt_exec_infos).items():
        print("\t{}: {}".format(k, v))

    # print("start dacecuda")
    # dace_exec_infos = run_horizontal_diffusion(
    #     niter=niter, domain=domain, backend=gt_backend.from_name("dacecuda")
    # )
    # print("start gtcuda")
    # gt_exec_infos = run_horizontal_diffusion(
    #     niter=niter, domain=domain, backend=gt_backend.from_name("gtcuda")
    # )
    # print("dace times:")
    # for k, v in summary(dace_exec_infos).items():
    #     print("\t{}: {}".format(k, v))
    # print("gt times:")
    # for k, v in summary(gt_exec_infos).items():
    #     print("\t{}: {}".format(k, v))
    # import gt4py.backend as gt_backend
    #
    # #############################################
